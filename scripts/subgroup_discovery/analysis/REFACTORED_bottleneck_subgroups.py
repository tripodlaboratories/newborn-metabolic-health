##########################################################################
## This script generates all subgroup results for bottleneck prediction ##
##########################################################################
# general imports
import argparse
import logging
import os
import pandas as pd
import numpy as np
import pickle
import yaml

# sklearn imports for evaluating results
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import roc_auc_score, average_precision_score

# Import our modules
import pysubgroup as ps
from biobank_project.subgroup_discovery import output
from biobank_project.subgroup_discovery.processing import SubgroupProcessor, FeatureTransformer
from biobank_project.subgroup_discovery.scoring import SubgroupScorer

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Perform subgroup discovery on bottleneck model outputs as health index",
        conflict_handler="resolve",
    )
    parser.add_argument(
        "-i",
        "--input_directory",
        default="./results/deep_mtl/neonatal_bottleneck_validation/ensemble_bottle_1/",
        help="Input experiment directory, should contain bottleneck.csv, valid_bottleneck.csv, and true_vals.csv",
    )
    parser.add_argument(
        "-o", "--output_directory", help="output directory to save results files"
    )
    parser.add_argument(
        "-t",
        "--tasks",
        type=str,
        default=None,
        help="Text file with tasks of interest, one task per line. These are the tasks evaluated for subgroup discovery.",
    )
    parser.add_argument(
        "--column_specification",
        type=str,
        default=None,
        help="Column specification YML containing keys: 'id', 'features', 'outcomes'",
    )
    return parser.parse_args()


def read_lines(file) -> list:
    with open(file) as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def load_data(results_dir, outcomes, id_col="row_id"):
    """Load bottleneck predictions and metadata"""
    logger.info("Loading data from %s", results_dir)

    # Read predictions and true values
    preds = pd.read_csv(f"{results_dir}/bottleneck.csv")
    val_preds = pd.read_csv(f"{results_dir}/valid_bottleneck.csv")
    true_vals = pd.read_csv(f"{results_dir}/true_vals.csv")

    # Read external validation data
    external_true_vals = pd.read_csv(
        "./data/processed/neonatal_conditions.csv", low_memory=False
    )

    # Standardize column names
    preds = preds.rename(columns={"Unnamed: 0": id_col})
    val_preds = val_preds.rename(columns={"Unnamed: 0": id_col})

    # Create pivoted predictions across iterations
    many_preds = preds.pivot_table(
        index=id_col, columns="iter", values="bottleneck_unit_0"
    )
    many_val_preds = val_preds.pivot_table(
        index=id_col, columns="iter", values="bottleneck_unit_0"
    )

    # Average predictions across iterations
    preds = preds.groupby([id_col])["bottleneck_unit_0"].mean()
    val_preds = val_preds.groupby([id_col])["bottleneck_unit_0"].mean()
    true_vals = true_vals.groupby([id_col])[outcomes].mean()

    # Subset external validation data
    external_true_vals = external_true_vals[outcomes]

    # Read metadata
    data = pd.read_csv("./data/processed/metadata.csv", low_memory=False)
    metabolite_labels = pd.read_csv("./config/metabolite_labels.csv")

    # Subset data to match predictions
    subset_data = data[data[id_col].isin(preds.index)]
    subset_data = subset_data.set_index(id_col)
    if "gdspid" in subset_data.columns:
        subset_data = subset_data.drop("gdspid", axis=1)

    # Subset validation data
    subset_val_data = data[data[id_col].isin(val_preds.index)]
    subset_val_data = subset_val_data.set_index(id_col)
    if "gdspid" in subset_val_data.columns:
        subset_val_data = subset_val_data.drop("gdspid", axis=1)

    # Get validation true values
    validation_true_vals = external_true_vals.loc[subset_val_data.index]

    # Verify alignment of indices
    assert (preds.index == subset_data.index).all(), (
        "Training predictions and data indices don't match"
    )
    assert (preds.index == true_vals.index).all(), (
        "Training predictions and true values indices don't match"
    )
    assert (val_preds.index == subset_val_data.index).all(), (
        "Validation predictions and data indices don't match"
    )
    assert (val_preds.index == validation_true_vals.index).all(), (
        "Validation predictions and true values indices don't match"
    )

    # Switch target to identify HEALTHY individuals (1 - outcome)
    true_vals = 1 - true_vals
    validation_true_vals = 1 - validation_true_vals

    return {
        "preds": preds,
        "val_preds": val_preds,
        "true_vals": true_vals,
        "validation_true_vals": validation_true_vals,
        "many_preds": many_preds,
        "many_val_preds": many_val_preds,
        "metadata": subset_data,
        "val_metadata": subset_val_data,
    }


def prepare_outcome_data(data_dict, outcome, outcomes):
    """Prepare data for a specific outcome"""
    # Get the clean names
    preds = data_dict["preds"]
    val_preds = data_dict["val_preds"]
    true_vals = data_dict["true_vals"]
    validation_true_vals = data_dict["validation_true_vals"]
    many_preds = data_dict["many_preds"]
    many_val_preds = data_dict["many_val_preds"]
    metadata = data_dict["metadata"]
    val_metadata = data_dict["val_metadata"]

    # Filter to remove examples with other conditions
    num_other_outcomes = len(outcomes) - 1
    keep = (
        true_vals[np.setdiff1d(outcomes, [outcome])].sum(axis=1) == num_other_outcomes
    ) | (true_vals[outcome] == 0)
    keep_val = (
        validation_true_vals[np.setdiff1d(outcomes, [outcome])].sum(axis=1)
        == num_other_outcomes
    ) | (validation_true_vals[outcome] == 0)

    # Subset data
    outcome_preds = preds.loc[keep]
    outcome_true_vals = true_vals.loc[keep, :]
    outcome_metadata = metadata.loc[keep, :]

    val_outcome_preds = val_preds.loc[keep_val]
    validation_outcome_true_vals = validation_true_vals.loc[keep_val, :]
    val_outcome_metadata = val_metadata.loc[keep_val, :]

    many_outcome_preds = many_preds.loc[keep]
    many_val_outcome_preds = many_val_preds.loc[keep_val]

    return {
        "outcome_preds": outcome_preds,
        "outcome_true_vals": outcome_true_vals,
        "outcome_metadata": outcome_metadata,
        "val_outcome_preds": val_outcome_preds,
        "validation_outcome_true_vals": validation_outcome_true_vals,
        "val_outcome_metadata": val_outcome_metadata,
        "many_outcome_preds": many_outcome_preds,
        "many_val_outcome_preds": many_val_outcome_preds,
    }


def prepare_quantile_data(data, feature_transformer=None, metabolite_columns=None):
    """Transform metabolite data to quantile features"""
    if feature_transformer is None:
        feature_transformer = FeatureTransformer(strategy='quantile')
    # Keep only non-missing values
    temp_data = data.loc[:, data.columns[data.isna().sum() == 0]]

    # Filter to keep only metabolite features if requested
    if metabolite_columns is not None:
        temp_data = temp_data[metabolite_columns]
    else:
        # Default to metabolite features with 'rc' in name
        keep_features = temp_data.columns.to_series().apply(
            lambda z: True if ("rc" in z) else False
        )
        temp_data = temp_data[temp_data.columns[keep_features]]

    # Transform metabolite columns to quantiles
    float_columns = temp_data.columns[temp_data.dtypes == "float64"]
    result = feature_transformer.create_quantile_features(
        data=temp_data,
        columns=float_columns,
        quantiles=[2, 3, 5]
    )

    return result


def discover_subgroups(data, outcome_column, preds_column, alpha, result_size, depth=4):
    """Use pysubgroup to discover subgroups"""
    # Create predicton target
    target = ps.PredictionTarget(
        data[outcome_column].to_numpy(),
        data[preds_column].to_numpy(),
        roc_auc_score,  # Default metric
    )

    # Create selectors for metabolite features
    is_metabolite = data.columns.to_series().apply(
        lambda z: True if "rc" in z else False
    )
    searchspace = ps.create_selectors(data[data.columns[is_metabolite]])

    # Configure task
    task = ps.SubgroupDiscoveryTask(
        data,
        target,
        searchspace,
        result_set_size=result_size,
        depth=depth,
        qf=ps.PredictionQFNumeric(a=alpha),
    )

    # Execute discovery
    results = ps.BeamSearch(beam_width=result_size).execute(task)
    return results


def evaluate_subgroups(
    data, subgroup_desc_list, outcome_column, preds_column, many_preds_df=None
):
    """Evaluate all subgroups against data"""
    processor = SubgroupProcessor(
        data=data,
        outcome_column=outcome_column,
        prediction_column=preds_column,
        multiple_iter_preds=many_preds_df,
    )

    # Add all subgroups
    processor.add_subgroups_from_list(subgroup_desc_list)

    # Get cumulative subgroups
    cumulative_subgroups = processor.get_cumulative_subgroups()

    # Evaluate metrics for all subgroups
    subgroup_metrics = processor.evaluate_subgroups(metrics=["auroc", "auprc"])

    # Evaluate metrics for cumulative subgroups
    cumulative_metrics = processor.evaluate_subgroups(metrics=["auroc", "auprc"])

    return subgroup_metrics, cumulative_metrics


def main(args):
    # Process arguments
    results_dir = args.input_directory
    output_dir = args.output_directory

    # Process either tasks or a column specification that includes tasks
    if args.tasks is not None:
        features = None
        outcomes = read_lines(args.tasks)
        id_col = "row_id"
    elif args.column_specification is not None:
        with open(args.column_specification, "r") as f:
            col_spec = yaml.safe_load(f)
        features = col_spec["features"]
        outcomes = col_spec["outcomes"]
        id_col = col_spec["id"]
    else:
        raise ValueError(
            "Must provide one of the following options: --tasks OR --column_specification"
        )

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data_dict = load_data(results_dir, outcomes, id_col)

    # Define subgroup discovery parameters
    outcome_order = outcomes
    evaluation_order = ["AUROC", "AVG Precision"]

    # Parameters per outcome
    subgroup_alphas_avg_prec = {
        "bpd_any": 0.059,
        "rop_any": 0.06,
        "ivh_any": 0.06,
        "nec_any": 0.025,
    }
    subgroup_sizes_avg_prec = {
        "bpd_any": 100,
        "rop_any": 300,
        "ivh_any": 100,
        "nec_any": 100,
    }

    subgroup_alphas_auroc = {
        "bpd_any": 0.0575,
        "rop_any": 0.073,
        "ivh_any": 0.0585,
        "nec_any": 0.085,
    }
    subgroup_sizes_auroc = {
        "bpd_any": 200,
        "rop_any": 300,
        "ivh_any": 100,
        "nec_any": 100,
    }

    subgroup_alphas_list = {
        "AVG Precision": subgroup_alphas_avg_prec,
        "AUROC": subgroup_alphas_auroc,
    }
    subgroup_sizes_list = {
        "AVG Precision": subgroup_sizes_avg_prec,
        "AUROC": subgroup_sizes_auroc,
    }
    evaluation_lists = {
        "AVG Precision": average_precision_score,
        "AUROC": roc_auc_score,
    }

    # Storage for top K predictions
    top_k_subgroup_predictions = []
    top_k_subgroup_preds_iters = []

    # Iterate through evaluation metrics
    for metric in evaluation_order:
        logger.info(f"Starting analysis using {metric}")

        subgroup_alphas = subgroup_alphas_list[metric]
        subgroup_sizes = subgroup_sizes_list[metric]
        evaluation_metric = evaluation_lists[metric]

        all_results = {}
        iter_results = {}

        # Analyze each outcome
        for outcome in outcome_order:
            logger.info(f"Starting analysis of {outcome}")

            # Prepare data for this outcome
            outcome_data = prepare_outcome_data(data_dict, outcome, outcomes)

            # Extract metabolite features
            temp_data = outcome_data["outcome_metadata"].copy()
            temp_data = temp_data.loc[:, temp_data.columns[temp_data.isna().sum() == 0]]

            # Keep only metabolite features
            keep_features = temp_data.columns.to_series().apply(
                lambda z: True if ("rc" in z) else False
            )
            temp_data = temp_data[temp_data.columns[keep_features]]

            # Transform to quantiles
            searchspace_data = prepare_quantile_data(
                temp_data, columns=temp_data.columns[temp_data.dtypes == "float64"]
            )

            # Do the same for validation data
            temp_val_data = outcome_data["val_outcome_metadata"].copy()
            temp_val_data = temp_val_data.loc[
                :, temp_val_data.columns[temp_val_data.isna().sum() == 0]
            ]
            keep_val_features = temp_val_data.columns.to_series().apply(
                lambda z: True if ("rc" in z) else False
            )
            temp_val_data = temp_val_data[temp_val_data.columns[keep_val_features]]
            searchspace_val_data = prepare_quantile_data(
                temp_val_data,
                columns=temp_val_data.columns[temp_val_data.dtypes == "float64"],
            )

            # Discover subgroups
            results = discover_subgroups(
                data=searchspace_data,
                outcome_column=outcome,
                preds_column=outcome_data["outcome_preds"],
                alpha=subgroup_alphas[outcome],
                result_size=subgroup_sizes[outcome],
            )

            # Extract subgroup descriptions
            subgroup_desc = results.to_dataframe()["subgroup"]

            # Create processors for training and validation data
            train_processor = SubgroupProcessor(
                data=searchspace_data,
                outcome_column=outcome,
                prediction_column=outcome_data["outcome_preds"],
                multiple_iter_preds=outcome_data["many_outcome_preds"],
            )

            # Add discovered subgroups
            for desc in subgroup_desc:
                train_processor.add_subgroup(desc)

            # Get cumulative subgroups
            train_cumulative = train_processor.get_cumulative_subgroups()

            # Create validation processor
            val_processor = SubgroupProcessor(
                data=searchspace_val_data,
                outcome_column=outcome,
                prediction_column=outcome_data["val_outcome_preds"],
                multiple_iter_preds=outcome_data["many_val_outcome_preds"],
            )

            # Add same subgroups to validation
            added_to_val = []
            for desc in subgroup_desc:
                if val_processor.add_subgroup(desc):
                    added_to_val.append(desc)

            # Process validation cumulative subgroups
            val_cumulative = val_processor.get_cumulative_subgroups()

            # Get training metrics
            train_metrics = [
                sg["scorer"].score_auprc(score_over_iters=True)
                for sg in train_processor.subgroups
            ]
            train_cumulative_metrics = [
                sg["scorer"].score_auprc(score_over_iters=True)
                for sg in train_cumulative
            ]

            # Get validation metrics
            val_metrics = [
                sg["scorer"].score_auprc(score_over_iters=True)
                for sg in val_processor.subgroups
            ]
            val_cumulative_metrics = [
                sg["scorer"].score_auprc(score_over_iters=True) for sg in val_cumulative
            ]

            # Calculate overall metrics on full datasets
            kfold_scorer = SubgroupScorer(
                sg_num=0,
                sg_description="full_dataset",
                sg_mask=np.full(len(searchspace_data), True),
                true_vals=outcome_data["outcome_true_vals"][outcome].values,
                preds=outcome_data["outcome_preds"].values,
                multiple_iter_preds=outcome_data["many_outcome_preds"],
            )

            val_scorer = SubgroupScorer(
                sg_num=0,
                sg_description="full_dataset",
                sg_mask=np.full(len(searchspace_val_data), True),
                true_vals=outcome_data["validation_outcome_true_vals"][outcome].values,
                preds=outcome_data["val_outcome_preds"].values,
                multiple_iter_preds=outcome_data["many_val_outcome_preds"],
            )

            # Get overall metrics
            kfold_auroc = kfold_scorer.score_auroc()
            kfold_auprc = kfold_scorer.score_auprc()
            kfold_auroc_mean, kfold_auroc_sd = kfold_scorer.score_auroc(
                score_over_iters=True
            )
            kfold_auprc_mean, kfold_auprc_sd = kfold_scorer.score_auprc(
                score_over_iters=True
            )

            val_auroc = val_scorer.score_auroc()
            val_auprc = val_scorer.score_auprc()
            val_auroc_mean, val_auroc_sd = val_scorer.score_auroc(score_over_iters=True)
            val_auprc_mean, val_auprc_sd = val_scorer.score_auprc(score_over_iters=True)

            # Generate random predictions for comparison
            np.random.seed(1234)
            rand_val_pred = np.random.uniform(
                0, 1, len(outcome_data["validation_outcome_true_vals"][outcome])
            )
            rand_pred = np.random.uniform(
                0, 1, len(outcome_data["outcome_true_vals"][outcome])
            )

            # Calculate random metrics
            rand_auroc = roc_auc_score(
                outcome_data["outcome_true_vals"][outcome], rand_pred
            )
            rand_precision, rand_recall, _ = precision_recall_curve(
                outcome_data["outcome_true_vals"][outcome], rand_pred
            )
            rand_auprc = auc(rand_recall, rand_precision)

            rand_val_auroc = roc_auc_score(
                outcome_data["validation_outcome_true_vals"][outcome], rand_val_pred
            )
            rand_val_precision, rand_val_recall, _ = precision_recall_curve(
                outcome_data["validation_outcome_true_vals"][outcome], rand_val_pred
            )
            rand_val_auprc = auc(rand_val_recall, rand_val_precision)

            # Calculate metrics at 20% data cutoff
            # Find subgroup close to 20% of the data
            train_percentages = [sg["percent"] for sg in train_cumulative]
            train_20pct_idx = np.abs(np.array(train_percentages) - 20).argmin()

            val_percentages = (
                [sg["percent"] for sg in val_cumulative] if val_cumulative else []
            )
            val_20pct_idx = (
                np.abs(np.array(val_percentages) - 20).argmin()
                if val_percentages
                else None
            )

            # Get metrics at 20% cutoff - training
            if train_20pct_idx is not None:
                sg_20pct = train_cumulative[train_20pct_idx]
                kfold_auroc_20 = sg_20pct["scorer"].score_auroc()
                kfold_auprc_20 = sg_20pct["scorer"].score_auprc()
                kfold_auroc_20_mean, kfold_auroc_20_sd = sg_20pct["scorer"].score_auroc(
                    score_over_iters=True
                )
                kfold_auprc_20_mean, kfold_auprc_20_sd = sg_20pct["scorer"].score_auprc(
                    score_over_iters=True
                )

                # Save selected subgroup predictions for detailed analysis
                mask_20pct = sg_20pct["mask"]
                preds_top_subgroups = outcome_data["outcome_preds"][mask_20pct]
                true_vals_top_subgroups = outcome_data["outcome_true_vals"].loc[
                    mask_20pct, outcome
                ]

                outcome_top_subgroups_df = pd.DataFrame(
                    {
                        "preds": preds_top_subgroups,
                        "true_vals": true_vals_top_subgroups,
                        "outcome": outcome,
                        "evaluation_metric": metric,
                        "dataset": "kfold_test",
                    }
                )
                outcome_top_subgroups_df["row_id"] = outcome_top_subgroups_df.index
                outcome_top_subgroups_df = outcome_top_subgroups_df.set_index("row_id")
                top_k_subgroup_predictions.append(outcome_top_subgroups_df)

                # Calculate random metrics for comparison
                rand_auroc_20 = roc_auc_score(
                    true_vals_top_subgroups, rand_pred[mask_20pct]
                )
                rand_precision, rand_recall, _ = precision_recall_curve(
                    true_vals_top_subgroups, rand_pred[mask_20pct]
                )
                rand_auprc_20 = auc(rand_recall, rand_precision)

                # Save individual iteration predictions
                preds_iters_top_subgroups = (
                    outcome_data["many_outcome_preds"]
                    .loc[mask_20pct]
                    .reset_index()
                    .melt(id_vars="row_id", value_name="preds")
                    .set_index("row_id")
                )
                top_subgroups_iters_df = pd.merge(
                    preds_iters_top_subgroups,
                    true_vals_top_subgroups.rename("true_vals"),
                    left_index=True,
                    right_index=True,
                )
                top_subgroups_iters_df["outcome"] = outcome
                top_subgroups_iters_df["evaluation_metric"] = metric
                top_subgroups_iters_df["dataset"] = "kfold_test"
                top_k_subgroup_preds_iters.append(top_subgroups_iters_df)
            else:
                # No subgroup near 20% found
                kfold_auroc_20 = np.nan
                kfold_auprc_20 = np.nan
                kfold_auroc_20_mean = np.nan
                kfold_auroc_20_sd = np.nan
                kfold_auprc_20_mean = np.nan
                kfold_auprc_20_sd = np.nan
                rand_auroc_20 = np.nan
                rand_auprc_20 = np.nan

            # Get metrics at 20% cutoff - validation
            if val_20pct_idx is not None and val_percentages:
                sg_20pct_val = val_cumulative[val_20pct_idx]
                val_auroc_20 = sg_20pct_val["scorer"].score_auroc()
                val_auprc_20 = sg_20pct_val["scorer"].score_auprc()
                val_auroc_20_mean, val_auroc_20_sd = sg_20pct_val["scorer"].score_auroc(
                    score_over_iters=True
                )
                val_auprc_20_mean, val_auprc_20_sd = sg_20pct_val["scorer"].score_auprc(
                    score_over_iters=True
                )

                # Save selected subgroup predictions
                mask_20pct_val = sg_20pct_val["mask"]
                val_preds_top_subgroups = outcome_data["val_outcome_preds"][
                    mask_20pct_val
                ]
                val_true_vals_top_subgroups = outcome_data[
                    "validation_outcome_true_vals"
                ].loc[mask_20pct_val, outcome]

                val_outcome_top_subgroups_df = pd.DataFrame(
                    {
                        "preds": val_preds_top_subgroups,
                        "true_vals": val_true_vals_top_subgroups,
                        "outcome": outcome,
                        "evaluation_metric": metric,
                        "dataset": "holdout_validation",
                    }
                )
                val_outcome_top_subgroups_df["row_id"] = (
                    val_outcome_top_subgroups_df.index
                )
                val_outcome_top_subgroups_df = val_outcome_top_subgroups_df.set_index(
                    "row_id"
                )
                top_k_subgroup_predictions.append(val_outcome_top_subgroups_df)

                # Calculate random metrics
                rand_val_auroc_20 = roc_auc_score(
                    val_true_vals_top_subgroups, rand_val_pred[mask_20pct_val]
                )
                rand_val_precision, rand_val_recall, _ = precision_recall_curve(
                    val_true_vals_top_subgroups, rand_val_pred[mask_20pct_val]
                )
                rand_val_auprc_20 = auc(rand_val_recall, rand_val_precision)

                # Save individual iteration predictions
                val_preds_iters_top_subgroups = (
                    outcome_data["many_val_outcome_preds"]
                    .loc[mask_20pct_val]
                    .reset_index()
                    .melt(id_vars="row_id", value_name="preds")
                    .set_index("row_id")
                )
                val_top_subgroups_iters_df = pd.merge(
                    val_preds_iters_top_subgroups,
                    val_true_vals_top_subgroups.rename("true_vals"),
                    left_index=True,
                    right_index=True,
                )
                val_top_subgroups_iters_df["outcome"] = outcome
                val_top_subgroups_iters_df["evaluation_metric"] = metric
                val_top_subgroups_iters_df["dataset"] = "holdout_validation"
                top_k_subgroup_preds_iters.append(val_top_subgroups_iters_df)
            else:
                # No validation subgroup near 20% found
                val_auroc_20 = np.nan
                val_auprc_20 = np.nan
                val_auroc_20_mean = np.nan
                val_auroc_20_sd = np.nan
                val_auprc_20_mean = np.nan
                val_auprc_20_sd = np.nan
                rand_val_auroc_20 = np.nan
                rand_val_auprc_20 = np.nan

            # Store results for this outcome
            iter_results[f"{outcome}"] = [
                kfold_auroc_mean,
                kfold_auroc_sd,
                val_auroc_mean,
                val_auroc_sd,
                kfold_auroc_20_mean,
                kfold_auroc_20_sd,
                val_auroc_20_mean,
                val_auroc_20_sd,
                kfold_auprc_mean,
                kfold_auprc_sd,
                val_auprc_mean,
                val_auprc_sd,
                kfold_auprc_20_mean,
                kfold_auprc_20_sd,
                val_auprc_20_mean,
                val_auprc_20_sd,
            ]

            # Prepare results dataframes for final storage
            train_result_data = []
            for i, sg in enumerate(train_processor.subgroups):
                train_result_data.append(
                    [
                        train_cumulative[i]["description"],  # total group
                        sg["description"],  # subgroup
                        train_cumulative[i]["size"],  # size
                        train_cumulative[i]["percent"],  # % data
                        sg["size"],  # subgroup size
                        i + 1,  # num_groups
                        train_cumulative[i]["scorer"].score_auprc(),  # AUPRC
                        sg["scorer"].score_auprc(),  # subgroup AUPRC
                        train_cumulative[i]["scorer"].score_auroc(),  # AUROC
                        sg["scorer"].score_auroc(),  # subgroup AUROC
                    ]
                )

            val_result_data = []
            for i, sg in enumerate(val_processor.subgroups):
                if i < len(val_cumulative):
                    val_result_data.append(
                        [
                            val_cumulative[i]["description"],  # total group
                            sg["description"],  # subgroup
                            val_cumulative[i]["size"],  # size
                            val_cumulative[i]["percent"],  # % data
                            sg["size"],  # subgroup size
                            i + 1,  # num_groups
                            val_cumulative[i]["scorer"].score_auprc(),  # AUPRC
                            sg["scorer"].score_auprc(),  # subgroup AUPRC
                            val_cumulative[i]["scorer"].score_auroc(),  # AUROC
                            sg["scorer"].score_auroc(),  # subgroup AUROC
                        ]
                    )

                    # Create dataframes from the results
            train_result_df = pd.DataFrame(
                train_result_data,
                columns=[
                    "total group",
                    "subgroup",
                    "size",
                    r"% data",
                    "subgroup size",
                    "num_groups",
                    "AUPRC",
                    "subgroup AUPRC",
                    "AUROC",
                    "subgroup AUROC",
                ],
            )

            val_result_df = pd.DataFrame(
                val_result_data,
                columns=[
                    "total group",
                    "subgroup",
                    "size",
                    r"% data",
                    "subgroup size",
                    "num_groups",
                    "AUPRC",
                    "subgroup AUPRC",
                    "AUROC",
                    "subgroup AUROC",
                ],
            )

            # Store results in output directory
            result_path = os.path.join(output_dir, f"{metric}_{outcome}")
            os.makedirs(result_path, exist_ok=True)

            # Write out training results
            train_result_df.to_csv(
                os.path.join(result_path, f"{outcome}_train_results.csv"), index=False
            )

            # Write out validation results
            val_result_df.to_csv(
                os.path.join(result_path, f"{outcome}_validation_results.csv"),
                index=False,
            )

            # Save overall metrics
            metrics_df = pd.DataFrame(
                {
                    "metric": [
                        "kfold_auroc",
                        "kfold_auprc",
                        "validation_auroc",
                        "validation_auprc",
                        "kfold_auroc_mean",
                        "kfold_auroc_sd",
                        "val_auroc_mean",
                        "val_auroc_sd",
                        "kfold_auprc_mean",
                        "kfold_auprc_sd",
                        "val_auprc_mean",
                        "val_auprc_sd",
                        "kfold_auroc_20",
                        "kfold_auprc_20",
                        "validation_auroc_20",
                        "validation_auprc_20",
                        "kfold_auroc_20_mean",
                        "kfold_auroc_20_sd",
                        "val_auroc_20_mean",
                        "val_auroc_20_sd",
                        "kfold_auprc_20_mean",
                        "kfold_auprc_20_sd",
                        "val_auprc_20_mean",
                        "val_auprc_20_sd",
                        "rand_auroc",
                        "rand_auprc",
                        "rand_val_auroc",
                        "rand_val_auprc",
                        "rand_auroc_20",
                        "rand_auprc_20",
                        "rand_val_auroc_20",
                        "rand_val_auprc_20",
                    ],
                    "value": [
                        kfold_auroc,
                        kfold_auprc,
                        val_auroc,
                        val_auprc,
                        kfold_auroc_mean,
                        kfold_auroc_sd,
                        val_auroc_mean,
                        val_auroc_sd,
                        kfold_auprc_mean,
                        kfold_auprc_sd,
                        val_auprc_mean,
                        val_auprc_sd,
                        kfold_auroc_20,
                        kfold_auprc_20,
                        val_auroc_20,
                        val_auprc_20,
                        kfold_auroc_20_mean,
                        kfold_auroc_20_sd,
                        val_auroc_20_mean,
                        val_auroc_20_sd,
                        kfold_auprc_20_mean,
                        kfold_auprc_20_sd,
                        val_auprc_20_mean,
                        val_auprc_20_sd,
                        rand_auroc,
                        rand_auprc,
                        rand_val_auroc,
                        rand_val_auprc,
                        rand_auroc_20,
                        rand_auprc_20,
                        rand_val_auroc_20,
                        rand_val_auprc_20,
                    ],
                }
            )

            metrics_df.to_csv(
                os.path.join(result_path, f"{outcome}_metrics.csv"), index=False
            )

            # Save discovered subgroup descriptions
            pd.DataFrame({"subgroup": subgroup_desc}).to_csv(
                os.path.join(result_path, f"{outcome}_subgroups.csv"), index=False
            )

            # Store results for later aggregation
            all_results[outcome] = {
                "train_results": train_result_df,
                "val_results": val_result_df,
                "metrics": metrics_df,
                "subgroups": subgroup_desc,
            }

            logger.info(f"Completed analysis for {outcome} using {metric}")

        # Save combined iteration results for this metric
        pd.DataFrame(iter_results).T.to_csv(
            os.path.join(output_dir, f"{metric}_iterations_summary.csv")
        )

    # Combine all top-k subgroup predictions
    if top_k_subgroup_predictions:
        top_k_df = pd.concat(top_k_subgroup_predictions)
        top_k_df.to_csv(os.path.join(output_dir, "top_k_subgroup_predictions.csv"))

    # Combine all iteration predictions for top-k subgroups
    if top_k_subgroup_preds_iters:
        top_k_iters_df = pd.concat(top_k_subgroup_preds_iters)
        top_k_iters_df.to_csv(
            os.path.join(output_dir, "top_k_subgroup_predictions_by_iteration.csv")
        )

    logger.info(f"All analyses complete. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()
    main(args)
