"""Evaluate subgroup discovery on an external set of data + model scores."""
import argparse
import joblib
import json
import logging
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pathlib import Path
import numpy as np

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import KBinsDiscretizer

import xlsxwriter
import pickle

#NOTE: altered pysubgroup package must be installed prior to analysis
#to install use the following command "pip install git+https://github.com/Teculos/pysubgroup.git@predictionQF"
import pysubgroup as ps

from biobank_project.subgroup_discovery.scoring import SubgroupScorer


# TODO: Refactor args + config
def get_args():
    parser = argparse.ArgumentParser(
        description='Perform subgroup discovery on health index outputs and new dataset',
        conflict_handler='resolve')
    parser.add_argument(
        '-i', '--input_predictions', default='./results/metabolic_health_index/',
        help='File containing metabolic health index outputs')
    parser.add_argument(
        '-t', '--true_values', default=None,
        help='File containing corresponding true values to the predictions file')
    parser.add_argument(
        '-m', '--metabolite_data',
        help='Input metabolite profile data for defining subgroup quantiles.')
    parser.add_argument(
        '-q', '--quantile_obj_dir', default=None,
        help='Directory with quantile definition objects for input metabolites')
    parser.add_argument(
        '--quantile_obj_pattern', default='discretizers.joblib',
        help='Pattern for identifying a quantile discretizer object')
    parser.add_argument(
        '-s', '--subgroups_obj_dir',
        help='Directory with subgroup definitions objects that have previously been defined')
    parser.add_argument(
        '--subgroups_obj_pattern', default='all_subgroups.joblib',
        help='Pattern for identifying an object containing all subgroups')
    parser.add_argument(
        '-c', '--config', default=None,
        help='JSON config file for subgroup discovery procedure.')
    parser.add_argument(
        '--id_col', default='b_ikn',
        help='ID column for predictions')
    parser.add_argument(
        '--health_index_col', default='health_index_score_0',
        help='column name that contains the health index score')
    parser.add_argument(
        '-o', '--output_directory',
        help='output directory to save results files')
    parser.add_argument(
        '-l', '--log_level', default='INFO', help='logger level')
    return parser.parse_args()

def main(args):
    input_preds_file = args.input_predictions
    true_vals_file = args.true_values
    metabolites_file = args.metabolite_data
    quantile_obj_dir = args.quantile_obj_dir
    quantile_obj_pattern = args.quantile_obj_pattern
    subgroups_obj_dir = args.subgroups_obj_dir
    subgroups_obj_pattern = args.subgroups_obj_pattern
    id_col = args.id_col
    health_index_col = args.health_index_col

    output_dir = args.output_directory
    config_file = args.config
    log_level = args.log_level

    # Set up logger
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    logging.basicConfig(level=numeric_level)
    logger = logging.getLogger('SubgroupDiscovery')

    with open(config_file) as f:
        config = json.load(f)

    # Read in results from the metabolic health index
    preds = pd.read_csv(input_preds_file)
    preds = preds.rename(columns={"Unnamed: 0": id_col})
    true_vals = pd.read_csv(true_vals_file).set_index(id_col)

    preds = preds.groupby([id_col])[health_index_col].mean()
    if 'iter' in true_vals.columns:
        true_vals = true_vals[true_vals.iter == 0].drop(
            columns=['fold', 'iter'])
    true_vals = true_vals.loc[preds.index, :]

    # Read in metabolite profiles to be used in defining quantiles
    metab = pd.read_csv(metabolites_file).set_index(id_col)
    metab_names = config.get('metabolite_list')

    # Align all predictions, true values, and metabolites by index
    preds_index = preds.index
    true_vals = true_vals.loc[preds.index]
    metab = metab.loc[preds.index]

    assert (preds.index == true_vals.index).all()
    assert (preds.index == metab.index).all()

    #order is bpd -> rop -> ivh -> nec
    default_order = ["bpd", "rop", "ivh", "nec"]
    order_from_config = config.get('outcome_order')
    outcome_order = order_from_config if order_from_config else default_order
    true_vals = true_vals[outcome_order]

    eval_order_from_config = config.get('evaluation_order')
    default_evaluation_order = ["AUROC", "AVG Precision"]
    evaluation_order = eval_order_from_config if eval_order_from_config else default_evaluation_order

    # Create a list of dataframes for the top K predictions from each subgroup discovery setting
    top_k_subgroup_predictions = []
    # This second list stores predictions from individual iterations
    top_k_subgroup_preds_iters = []

    for metric in evaluation_order:
        logger.info(f'Evaluating subgroup discovery with metric: {metric}')

        all_results = {}
        iter_results = {}

        #start of loop to collect all data
        for outcome in outcome_order:
            logger.info(f'Beginning subgroup discovery procedure for: {outcome}')
            targ = outcome
            targ_true_vals = true_vals[outcome]

            # Find existing artifacts
            if quantile_obj_dir is not None:
                quantile_obj_dir = Path(quantile_obj_dir)
                discretizer_match = quantile_obj_dir.glob(f'*{metric}*{outcome}*{quantile_obj_pattern}')
                discretizer_files = list(discretizer_match)
                if len(discretizer_files) == 1:
                    discretizer_file = str(discretizer_files[0])
                    logger.info(f'Matching discretizer object: {discretizer_file}')
                    saved_discretizers = joblib.load(discretizer_file)
                else:
                    logger.error('Discretizers file was not matched successfully based on pattern.')
                    saved_discretizers = None
            else:
                saved_discretizers = None

            subgroup_file_match = Path(subgroups_obj_dir).glob(f'*{metric}*{outcome}*{subgroups_obj_pattern}')
            subgroup_files = list(subgroup_file_match)
            if len(subgroup_files) == 1:
                subgroup_file = str(subgroup_files[0])
                logger.info(f'Matching object: {subgroup_file}')
                saved_subgroups = joblib.load(subgroup_file)
            else:
                logger.error('Matching subgroup file not found.')

            # Extra string appended to columns to differentiate outcomes from subgroup discovery results
            col_annotation = "_sgdisc"

            # NOTE: limit to True healthy controls (removing controls with positive co-outcomes)
            other_outcomes = np.setdiff1d(outcome_order, targ)
            in_true_controls = (true_vals[other_outcomes].sum(axis=1) == 0) & (true_vals[targ] == 0)
            in_analysis_set = (in_true_controls) | (true_vals[targ] == 1)

            # NOTE: Since we are interested in identifying HEALTHY individuals
            # The target for prediction will be switched to healthy obs
            healthy_true_vals = 1 - true_vals # this along with a change in the 'in_analysis_set' vector is the only change

            metab_data = metab[metab_names]
            searchspace_input = metab_data[metab_data.columns.values[metab_data.isna().sum() == 0]].copy()
            searchspace_input = searchspace_input.loc[preds.index]

            #constructing list of demographic and metabolomic features to keep
            # NOTE: Currently only filters on metabolite columns, but extension
            # to other demographics would require different logic
            # e.g., {columns are in metabolites} | searchspace_input.columns.isin(demographics_features)
            in_analysis_set_features = (searchspace_input.columns.isin(metab_names))
            searchspace_input = searchspace_input[searchspace_input.columns[in_analysis_set_features]]

            #compile list of features which need to be transformed into quantiles
            # NOTE: Categorical features need to be protected from quantile transformation
            to_transform = searchspace_input.columns.isin(metab_names) | searchspace_input.apply(is_numeric_dtype)
            cols_to_transform = searchspace_input.columns[to_transform]
            transform_input = searchspace_input[cols_to_transform]

            # Either we define de novo quantiles or use a saved KBinsDiscretizer object
            if saved_discretizers:
                discretizers = saved_discretizers
            else:
                searchspace_quantiles = config.get('searchspace_quantiles')
                discretizers = {
                    q: KBinsDiscretizer(n_bins=q, encode='ordinal')
                    for q in searchspace_quantiles}

            quantile_dfs = []
            for quantile, disc in discretizers.items():
                quantile_output = disc.fit_transform(transform_input)
                quantile_df = pd.DataFrame(
                    quantile_output.astype(int),
                    columns=[f'{c}_q-{quantile}' for c in transform_input.columns],
                    index=transform_input.index)
                quantile_dfs.append(quantile_df)

            searchspace_data = pd.concat(quantile_dfs, axis=1)
            unmod_cols = searchspace_input.loc[:, ~to_transform]
            if len(unmod_cols.columns) != 0:
                searchspace_data = pd.concat([searchspace_data, unmod_cols], axis=1)

            is_metabolite = searchspace_data.columns.str.replace(
                r'_q.*$', '').isin(metab_names)

            # TODO: NOTE: This is the previous logic in another script regarding how subgroup discovery is initialized
            # NOTE: Here we will use the previously saved subgroup discovery object


            # INIT subgroup discovery objects and procedure
            # target = ps.PredictionTarget(targ_true_vals.to_numpy(), outcome_preds.to_numpy(), evaluation_metric)
            # searchspace = ps.create_selectors(searchspace_data[searchspace_data.columns[is_metabolite]])
            # task = ps.SubgroupDiscoveryTask(
            #     searchspace_data,
            #     target,
            #     searchspace,
            #     result_set_size=subgroup_sizes[outcome],
            #     depth=3,
            #     qf=ps.PredictionQFNumeric(a=subgroup_alphas[outcome])
            # )

            # results = ps.BeamSearch(beam_width=subgroup_sizes[outcome]).execute(task)

            # # Iterate over subgroups, collecting performance metrics
            # subgroup_desc_df = results.to_dataframe()

            # Iterate over the subgroups, finding the number of individuals in the subgroup and score
            # Book-keeping for a continuously growing merging of subgroups
            summary_stats = []
            total_merge_description = ''
            total_merge_mask = np.full((len(searchspace_data.index)), False)
            for sg_num, sg in enumerate(saved_subgroups):
                sg_quality, sg_description, qf = sg
                sg_mask, sg_size = ps.get_cover_array_and_size(sg_description, data=searchspace_data)

                # To be used to calculate statistics on a merged top 1:current_subgroup (as a running total)
                total_merge_mask = np.logical_or(total_merge_mask, sg_mask)

                # Iterating through subgroups to add custom stats (i.e., AUROC and AUPRC)
                runtotal_scorer = SubgroupScorer(
                    'Total Merged Subgroup', f'Merged from {sg_num} subgroups', total_merge_mask,
                    true_vals=targ_true_vals, preds=preds)
                AUROC = runtotal_scorer.score_auroc()
                AUPRC = runtotal_scorer.score_auprc()
                # Next calculate the AUPRC in the subgroup
                subgroup_scorer = SubgroupScorer(
                    sg_num, sg_description, sg_mask, true_vals=targ_true_vals, preds=preds)
                subgroup_AUROC = subgroup_scorer.score_auroc()
                subgroup_AUPRC = subgroup_scorer.score_auprc()

                # Create ongoing subgroup descriptions
                if len(total_merge_description) == 0:
                    total_merge_description = str(sg_description)
                else:
                    total_merge_description = str(sg_description) + "-OR-" + total_merge_description
                summary_stats.append({
                    'total_subgroup_merge': total_merge_description,
                    'subgroup': str(sg_description),
                    'size': total_merge_mask.sum(),
                    r'%data': total_merge_mask.sum() / len(searchspace_data.index),
                    "subgroup size": sg_size,
                    "subgroup number": sg_num,
                    "AUPRC": AUPRC,
                    "subgroup AUPRC": subgroup_AUPRC,
                    "AUROC": AUROC,
                    "subgroup AUROC": subgroup_AUROC})

            # Gather results from all subgroups
            subgroup_results_df = pd.DataFrame(summary_stats)

            #compile metrics of performance WITHOUT any subgroup masking
            nomask_scorer = SubgroupScorer(
                'No Subgroup Mask', 'Entire Dataset', np.full(len(preds), True),
                true_vals=targ_true_vals, preds=preds)
            kfold_AUROC = nomask_scorer.score_auroc()
            kfold_AUPRC = nomask_scorer.score_auprc()

            # SECTION: Comparison to random predictions
            #create random vectors on the test and validation data
            np.random.seed(1234)
            rand_pred = np.random.uniform(0,1,len(true_vals[targ]))
            rand_scorer = SubgroupScorer(
                'Random', 'Evaluation against random predictions',
                np.full(len(targ_true_vals), True),
                true_vals=targ_true_vals, preds=rand_pred)
            rand_AUROC = rand_scorer.score_auroc()
            rand_AUPRC = rand_scorer.score_auprc()


            # Usually the top 20 percentile of data
            select = (subgroup_results_df[r"%data"] * 100)
            select_index = select.index[select == min(select, key=lambda x:abs(x-20))][0]

            # Iterate over subgroups, collecting performance metrics ONLY in the top
            # 20% subgroup
            summary_stats = []
            total_merge_description = ''
            total_merge_mask = np.full((len(searchspace_data.index)), False)
            for sg_num, sg in enumerate(saved_subgroups):
                sg_quality, sg_description, qf = sg
                sg_mask, sg_size = ps.get_cover_array_and_size(sg_description, data=searchspace_data)
                total_merge_mask = np.logical_or(total_merge_mask, sg_mask)

                if sg_num == select_index:
                    preds_top_subgroups = preds[total_merge_mask]
                    true_vals_top_subgroups = targ_true_vals[total_merge_mask]
                    random_preds_top_subgroups = rand_pred[total_merge_mask]
                    assert sorted(preds_top_subgroups.index) == sorted(true_vals_top_subgroups.index)
                    outcome_top_subgroups_df = pd.DataFrame.from_dict(
                        {preds_top_subgroups.index.name: preds_top_subgroups.index,
                         'preds': preds_top_subgroups,
                         'true_vals': true_vals_top_subgroups,
                         'outcome': targ,
                         'evaluation_metric': metric,
                         'dataset': 'kfold_test'}).set_index(preds_top_subgroups.index.name)
                    top_k_subgroup_predictions.append(outcome_top_subgroups_df)

                    top20_scorer = SubgroupScorer(
                        'Top 20', 'Top 20 Percent of Data', total_merge_mask,
                        true_vals=true_vals_top_subgroups, preds=preds_top_subgroups)
                    kfold_AUROC_20 = top20_scorer.score_auroc()
                    ROC_tuple_20 = top20_scorer._score_overall_auroc(return_tuple=True)
                    kfold_AUPRC_20 = top20_scorer.score_auprc()
                    PR_tuple_20 = top20_scorer._score_overall_auprc(return_tuple=True)

                    top20_rand_scorer = SubgroupScorer(
                       'Top 20 Random', r'Random Predictions on top 20%data', total_merge_mask,
                        true_vals=true_vals_top_subgroups, preds=random_preds_top_subgroups)
                    rand_AUROC_20 = top20_rand_scorer.score_auroc()
                    rand_AUPRC_20 = top20_rand_scorer.score_auprc()
                    break

            # No repeated KFold iter results in the validation case
            iter_results[targ+col_annotation] = [
                None, None, None, None,
                None, None, None, None,
                None, None, None, None,
                None, None, None, None]
            all_results[targ+col_annotation] = [
                subgroup_results_df, None,
                kfold_AUROC, kfold_AUPRC, None, None,
                PR_tuple_20, None, kfold_AUPRC_20, None,
                ROC_tuple_20, None, kfold_AUROC_20, None,
                rand_AUROC, rand_AUPRC, None, None,
                rand_AUROC_20, rand_AUPRC_20, None, None]

        ###################
        # Writing outputs #
        ###################
        #save to file
        with open(output_dir + metric + "_bottleneck_results.pkl", "wb") as f:
            pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

        logger.info(f'saved .pkl results: to {output_dir}')

        #initialize writer
        # nan_inf_to_errors option allows nan to appear as #NUM! and inf as #DIV/0!
        workbook = xlsxwriter.Workbook(
            output_dir + metric + "_bottleneck_results.xlsx",
            {'nan_inf_to_errors': True})
        worksheet_baseline = workbook.add_worksheet("baseline")
        worksheet_baseline_mean = workbook.add_worksheet("Mean+SD Across Preds")
        worksheet_baseline_rand = workbook.add_worksheet("rand baseline")
        worksheet_20 = workbook.add_worksheet("baseline @ 20% Data")
        worksheet_20_mean = workbook.add_worksheet("Mean+SD Across Preds @ 20% Data")
        worksheet_20_rand = workbook.add_worksheet("rand baseline @ 20% Data")

        for outcome in outcome_order:
            targ=outcome
            col_annotation = "_sgdisc"

            worksheet_train = workbook.add_worksheet(outcome+"-train")
            worksheet_val = workbook.add_worksheet(outcome+"-val")
            worksheet_pr = workbook.add_worksheet(outcome+"-Kfold PR @ 20%")
            worksheet_roc = workbook.add_worksheet(outcome+"-Kfold ROC @ 20%")
            worksheet_pr_val = workbook.add_worksheet(outcome+"-Val PR @ 20%")
            worksheet_roc_val = workbook.add_worksheet(outcome+"-Val ROC @ 20%")

            subgroup_results = all_results[targ+col_annotation][0]
            subgroup_val_results = all_results[targ+col_annotation][1]

            kfold_AUROC = all_results[targ+col_annotation][2]
            kfold_AUPRC = all_results[targ+col_annotation][3]

            val_AUROC = all_results[targ+col_annotation][4]
            val_AUPRC = all_results[targ+col_annotation][5]

            (precision_20, recall_20, thresholds_20) = all_results[targ+col_annotation][6]
            (precision_val_20, recall_val_20, thresholds_val_20) = (None, None, None)
            #(precision_val_20, recall_val_20, thresholds_val_20) = all_results[targ+col_annotation][7]

            kfold_20_AUPRC = all_results[targ+col_annotation][8]
            val_20_AUPRC = all_results[targ+col_annotation][9]

            (fpr_20, tpr_20, thresholds_20) = all_results[targ+col_annotation][10]
            (fpr_val_20, tpr_val_20, thresholds_val_20) = (None, None, None)
            #(fpr_val_20, tpr_val_20, thresholds_val_20) = all_results[targ+col_annotation][11]

            kfold_20_AUROC = all_results[targ+col_annotation][12]
            val_20_AUROC = all_results[targ+col_annotation][13]


            worksheet_baseline.write(0,0, "outcome")
            worksheet_baseline.write(0,1, "kfold AUROC")
            worksheet_baseline.write(0,2, "kfold AUPRC")
            worksheet_baseline.write(0,3, "val AUROC")
            worksheet_baseline.write(0,4, "val AUPRC")

            worksheet_baseline.write(outcome_order.index(outcome)+1, 0, outcome)
            worksheet_baseline.write(outcome_order.index(outcome)+1, 1, kfold_AUROC)
            worksheet_baseline.write(outcome_order.index(outcome)+1, 2, kfold_AUPRC)
            worksheet_baseline.write(outcome_order.index(outcome)+1, 3, val_AUROC)
            worksheet_baseline.write(outcome_order.index(outcome)+1, 4, val_AUPRC)


            worksheet_20.write(0,0, "outcome")
            worksheet_20.write(0,1, "kfold AUROC")
            worksheet_20.write(0,2, "kfold AUPRC")
            worksheet_20.write(0,3, "val AUROC")
            worksheet_20.write(0,4, "val AUPRC")

            worksheet_20.write(outcome_order.index(outcome)+1, 0, outcome)
            worksheet_20.write(outcome_order.index(outcome)+1, 1, kfold_20_AUROC)
            worksheet_20.write(outcome_order.index(outcome)+1, 2, kfold_20_AUPRC)
            worksheet_20.write(outcome_order.index(outcome)+1, 3, val_20_AUROC)
            worksheet_20.write(outcome_order.index(outcome)+1, 4, val_20_AUPRC)


            kfold_AUROC_mean = iter_results[targ+col_annotation][0]
            kfold_AUROC_sd = iter_results[targ+col_annotation][1]
            val_AUROC_mean = iter_results[targ+col_annotation][2]
            val_AUROC_sd = iter_results[targ+col_annotation][3]

            kfold_AUROC_mean_20 = iter_results[targ+col_annotation][4]
            kfold_AUROC_sd_20 = iter_results[targ+col_annotation][5]
            val_AUROC_mean_20 = iter_results[targ+col_annotation][6]
            val_AUROC_sd_20 = iter_results[targ+col_annotation][7]

            kfold_AUPRC_mean = iter_results[targ+col_annotation][8]
            kfold_AUPRC_sd = iter_results[targ+col_annotation][9]
            val_AUPRC_mean = iter_results[targ+col_annotation][10]
            val_AUPRC_sd = iter_results[targ+col_annotation][11]

            kfold_AUPRC_mean_20 = iter_results[targ+col_annotation][12]
            kfold_AUPRC_sd_20 = iter_results[targ+col_annotation][13]
            val_AUPRC_mean_20 = iter_results[targ+col_annotation][14]
            val_AUPRC_sd_20 = iter_results[targ+col_annotation][15]

            worksheet_20_mean.write(0,0, "outcome")
            worksheet_20_mean.write(0,1, "kfold AUROC")
            worksheet_20_mean.write(0,2, "kfold AUPRC")
            worksheet_20_mean.write(0,3, "val AUROC")
            worksheet_20_mean.write(0,4, "val AUPRC")

            worksheet_20_mean.write(0,6, "kfold AUROC SD")
            worksheet_20_mean.write(0,7, "kfold AUPRC SD")
            worksheet_20_mean.write(0,8, "val AUROC SD")
            worksheet_20_mean.write(0,9, "val AUPRC SD")


            worksheet_20_mean.write(outcome_order.index(outcome)+1, 0, outcome)
            worksheet_20_mean.write(outcome_order.index(outcome)+1, 1, kfold_AUROC_mean_20)
            worksheet_20_mean.write(outcome_order.index(outcome)+1, 2, kfold_AUPRC_mean_20)
            worksheet_20_mean.write(outcome_order.index(outcome)+1, 3, val_AUROC_mean_20)
            worksheet_20_mean.write(outcome_order.index(outcome)+1, 4, val_AUPRC_mean_20)

            worksheet_20_mean.write(outcome_order.index(outcome)+1, 6, kfold_AUROC_sd_20)
            worksheet_20_mean.write(outcome_order.index(outcome)+1, 7, kfold_AUPRC_sd_20)
            worksheet_20_mean.write(outcome_order.index(outcome)+1, 8, val_AUROC_sd_20)
            worksheet_20_mean.write(outcome_order.index(outcome)+1, 9, val_AUPRC_sd_20)

            worksheet_baseline_mean.write(0,0, "outcome")
            worksheet_baseline_mean.write(0,1, "kfold AUROC")
            worksheet_baseline_mean.write(0,2, "kfold AUPRC")
            worksheet_baseline_mean.write(0,3, "val AUROC")
            worksheet_baseline_mean.write(0,4, "val AUPRC")

            worksheet_baseline_mean.write(0,6, "kfold AUROC SD")
            worksheet_baseline_mean.write(0,7, "kfold AUPRC SD")
            worksheet_baseline_mean.write(0,8, "val AUROC SD")
            worksheet_baseline_mean.write(0,9, "val AUPRC SD")

            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 0, outcome)
            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 1, kfold_AUROC_mean)
            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 2, kfold_AUPRC_mean)
            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 3, val_AUROC_mean)
            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 4, val_AUPRC_mean)

            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 6, kfold_AUROC_sd)
            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 7, kfold_AUPRC_sd)
            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 8, val_AUROC_sd)
            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 9, val_AUPRC_sd)


            rand_AUROC = all_results[targ+col_annotation][14]
            rand_AUPRC = all_results[targ+col_annotation][15]

            rand_val_AUROC = all_results[targ+col_annotation][16]
            rand_val_AUPRC = all_results[targ+col_annotation][17]


            rand_AUROC_20 = all_results[targ+col_annotation][18]
            rand_AUPRC_20 = all_results[targ+col_annotation][19]

            rand_val_AUROC_20 = all_results[targ+col_annotation][20]
            rand_val_AUPRC_20 = all_results[targ+col_annotation][21]

            worksheet_20_rand.write(0,0, "outcome")
            worksheet_20_rand.write(0,1, "kfold AUROC")
            worksheet_20_rand.write(0,2, "kfold AUPRC")
            worksheet_20_rand.write(0,3, "val AUROC")
            worksheet_20_rand.write(0,4, "val AUPRC")

            worksheet_20_rand.write(outcome_order.index(outcome)+1, 0, outcome)
            worksheet_20_rand.write(outcome_order.index(outcome)+1, 1, rand_AUROC_20)
            worksheet_20_rand.write(outcome_order.index(outcome)+1, 2, rand_AUPRC_20)
            worksheet_20_rand.write(outcome_order.index(outcome)+1, 3, rand_val_AUROC_20)
            worksheet_20_rand.write(outcome_order.index(outcome)+1, 4, rand_val_AUPRC_20)

            worksheet_baseline_rand.write(0,0, "outcome")
            worksheet_baseline_rand.write(0,1, "kfold AUROC")
            worksheet_baseline_rand.write(0,2, "kfold AUPRC")
            worksheet_baseline_rand.write(0,3, "val AUROC")
            worksheet_baseline_rand.write(0,4, "val AUPRC")

            worksheet_baseline_rand.write(outcome_order.index(outcome)+1, 0, outcome)
            worksheet_baseline_rand.write(outcome_order.index(outcome)+1, 1, rand_AUROC)
            worksheet_baseline_rand.write(outcome_order.index(outcome)+1, 2, rand_AUPRC)
            worksheet_baseline_rand.write(outcome_order.index(outcome)+1, 3, rand_val_AUROC)
            worksheet_baseline_rand.write(outcome_order.index(outcome)+1, 4, rand_val_AUPRC)

            #adding column titles
            col_num=0
            row_num=0
            for col in subgroup_results.columns:
                temp = worksheet_train.write(row_num, col_num, col)
                col_num = col_num + 1

            #adding subgroup result vectors
            col_num=0
            for col in subgroup_results.columns:
                row_num = 1
                for val in subgroup_results[col]:
                    if pd.isna(val): #nan check
                        temp = worksheet_train.write(row_num, col_num, "nan")
                    else:
                        temp = worksheet_train.write(row_num, col_num, val)
                    row_num = row_num + 1
                col_num = col_num + 1

            #adding precision recall data
            worksheet_pr.write(0,0, "Precision KFold")
            worksheet_pr.write(0,1, "Recall KFold")
            worksheet_pr_val.write(0,0, "Precision Val")
            worksheet_pr_val.write(0,1, "Recall Val")
            if not pd.isna(precision_20).all() and not pd.isna(recall_20).all():
                for row_num, (prec, rec) in enumerate(zip(precision_20, recall_20)):
                    worksheet_pr.write(row_num+1,0,prec)
                    worksheet_pr.write(row_num+1,1,rec)

            # adding ROC tpr and fpr axes
            worksheet_roc.write(0,0, "TPR KFold")
            worksheet_roc.write(0,1, "FPR KFold")
            worksheet_roc_val.write(0,0, "TPR Val")
            worksheet_roc_val.write(0,1, "FPR Val")
            if not pd.isna(tpr_20).all() and not pd.isna(fpr_20).all():
                for row_num, (tpr, fpr) in enumerate(zip(tpr_20, fpr_20)):
                    worksheet_roc.write(row_num+1,0,tpr)
                    worksheet_roc.write(row_num+1,1,fpr)

        workbook.close()
        logger.info(f'Writing .csv to {output_dir}')

    # Save top K subgroup predictions
    top_k_subgroup_predictions = pd.concat(top_k_subgroup_predictions)
    top_k_subgroup_predictions.to_csv(
        output_dir + 'top_k_subgroup_predictions.csv')


if __name__ == '__main__':
    args = get_args()
    main(args)
