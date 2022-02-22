# general imports
import argparse
import logging
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np

# sklearn imports for evaluating results
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

#importing some utilities functions
import xlsxwriter
import pickle

#NOTE: altered pysubgroup package must be installed prior to analysis
#to install use the following command "pip install git+https://github.com/Teculos/pysubgroup.git@predictionQF"
import pysubgroup as ps

from biobank_project.subgroup_discovery.scoring import SubgroupScorer

#setting printing parameters so subgroup descriptions actually display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 250)
pd.set_option('display.max_colwidth', 150)

def get_args():
    parser = argparse.ArgumentParser(
        description='Perform subgroup discovery on bottleneck model outputs as health index',
        conflict_handler='resolve')
    parser.add_argument(
        '-i', '--input_directory', default='./results/deep_mtl/neonatal_bottleneck_validation/ensemble_bottle_1/',
        help='Input experiment directory, should contain bottleneck.csv, valid_bottleneck.csv, and true_vals.csv')
    parser.add_argument(
        '--validation_metabolites',
        default='./external_validation/mednax/processed/mednax_metabolites_cal_names.csv',
        help='Metabolites file corresponding to validation data, metabolites must match.')
    parser.add_argument(
        '--validation_predictions',
        default="./results/external_validation/mednax/health_index_output.csv",
        help='Filename for the predictions file on the validation dataset')
    parser.add_argument(
        '--validation_true_vals',
        default='./external_validation/mednax/processed/mednax_outcomes.csv',
        help='Corresponding true values file for the validation dataset.')
    parser.add_argument(
        '--validation_id', default='QuestionsRCode',
        help='ID column for the validation predictions')
    parser.add_argument(
        '-o', '--output_directory',
        help='output directory to save results files')
    parser.add_argument(
        '-l', '--log_level', default='INFO', help='logger level')
    return parser.parse_args()


def main(args):
    results_dir = args.input_directory
    valid_metab_file = args.validation_metabolites
    valid_preds_file = args.validation_predictions
    valid_true_vals_file = args.validation_true_vals
    output_dir = args.output_directory
    val_index_col = args.validation_id
    log_level = args.log_level

    # Set up logger
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    logging.basicConfig(level=numeric_level)
    logger = logging.getLogger('SubgroupDiscovery')

    # read in previous outputs from the bottleneck layer
    preds = pd.read_csv(results_dir + "bottleneck.csv")
    preds = preds.rename(columns={"Unnamed: 0":"row_id"})
    true_vals = pd.read_csv(results_dir + "true_vals.csv").set_index('row_id')

    # Read in Mednax predictions and true values
    # Read in Mednax metabolites
    val_preds = pd.read_csv(valid_preds_file).set_index(val_index_col)
    valid_true_vals = pd.read_csv(valid_true_vals_file).set_index(val_index_col)
    valid_metab = pd.read_csv(valid_metab_file).set_index(val_index_col)

    #read in raw data to get actual response for validation data, not currently included in prediction .csv's
    cal_biobank_data = pd.read_csv("./data/processed/neonatal_conditions.csv", low_memory=False)

    #maintain predictions across individual model runs to calculate Mean + SD
    preds_over_iters = preds.copy().pivot_table(index="row_id",columns="iter",values="bottleneck_unit_0")
    preds = preds.groupby(["row_id"])["bottleneck_unit_0"].mean()
    if 'iter' in true_vals.columns:
        true_vals = true_vals[true_vals.iter == 0].drop(
            columns=['fold', 'iter'])
    true_vals = true_vals.loc[preds.index, :]

    # Handling for validation predictions, which may not be over the same iters
    val_prediction_col = val_preds.drop(
            columns=['fold', 'iter', val_index_col],
            errors='ignore').columns[0]
    if 'iter' not in val_preds.columns:
        val_preds_over_iters = val_preds.copy()
        val_preds_over_iters['iter'] = 0
        val_preds_over_iters = val_preds_over_iters.pivot_table(
            index=val_index_col, columns="iter",
            values=val_prediction_col)
    else:
        val_preds_over_iters = val_preds_over_iters.pivot_table(
            index=val_index_col, columns="iter",
            values=val_prediction_col)
    val_preds = val_preds.groupby(val_index_col)[val_prediction_col].mean()

    #collapse all outcomes to patients x outcomes dataframe
    external_true_vals = cal_biobank_data[["nec_any","rop_any","bpd_any", "ivh_any"]]
    metabolite_labels = pd.read_csv("./config/metabolite_labels.csv")

    # Read in the California metabolite labels
    with open('./config/expected_metabolite_order.txt') as f:
        cal_metabolites = [l.strip() for l in f.readlines()]

    # Read in metadata
    metadata = pd.read_csv("./data/processed/metadata.csv", low_memory=False)

    validation_true_vals = valid_true_vals.loc[val_preds.index]

    #check that all indices are the same
    assert (preds.index == true_vals.index).all()
    assert (preds.index == preds_over_iters.index).all()
    assert (val_preds.index == validation_true_vals.index).all()

    #order is bpd -> rop -> ivh -> nec
    outcome_order = ["bpd", "rop", "ivh", "nec"]
    evaluation_order = ["AUROC", "AVG Precision"]

    subgroup_alphas_avg_prec = {"bpd":0.059, "rop":0.06, "ivh":0.06, "nec":0.025}
    subgroup_sizes_avg_prec = {"bpd":100, "rop":300, "ivh":100, "nec":100}

    subgroup_alphas_auroc = {"bpd":0.0575, "rop":0.073, "ivh":0.0585, "nec":0.085}
    subgroup_sizes_auroc = {"bpd":200, "rop":300, "ivh":100, "nec":100}

    subgroup_alphas_list = {"AVG Precision":subgroup_alphas_avg_prec, "AUROC":subgroup_alphas_auroc}
    subgroup_sizes_list = {"AVG Precision":subgroup_sizes_avg_prec, "AUROC":subgroup_sizes_auroc}
    evaluation_lists = {"AVG Precision":average_precision_score, "AUROC":roc_auc_score}

    # Create a list of dataframes for the top K predictions from each subgroup discovery setting
    top_k_subgroup_predictions = []
    # This second list stores predictions from individual iterations
    top_k_subgroup_preds_iters = []

    for metric in evaluation_order:
        logger.info(f'Starting subgroup discovery with metric: {metric}')

        subgroup_alphas = subgroup_alphas_list[metric]
        subgroup_sizes = subgroup_sizes_list[metric]
        evaluation_metric = evaluation_lists[metric]

        all_results = {}
        iter_results = {}

        #start of loop to collect all data
        for outcome in outcome_order:
            logger.info(f'Beginning subgroup discovery procedure for: {outcome}')

            targ = outcome+"_any"
            # Extra string appended to columns to differentiate outcomes from subgroup discovery results
            # TODO: Testing: Does this mangle any of the columns
            col_annotation = "_sgdisc"

            # NOTE: limit to True healthy controls (removing controls with positive co-outcomes)
            outcome_labels = ["nec_any","rop_any","bpd_any","ivh_any"]
            other_outcomes = np.setdiff1d(outcome_labels, targ)
            in_true_controls = (true_vals[other_outcomes].sum(axis=1) == 0) & (true_vals[targ] == 0)
            in_analysis_set = (in_true_controls) | (true_vals[targ] == 1)
            in_true_controls_val = (validation_true_vals[other_outcomes].sum(axis=1) == 0) & (validation_true_vals[targ] == 0)
            in_analysis_set_val = (in_true_controls_val) | (validation_true_vals[targ] == 1)

            # NOTE: Since we are interested in identifying HEALTHY individuals
            # The target for prediction will be switched to healthy obs
            healthy_true_vals = 1 - true_vals # this along with a change in the 'in_analysis_set' vector is the only change
            healthy_validation_true_vals = 1 - validation_true_vals[outcome_labels]

            #k-fold
            outcome_preds = preds.loc[in_analysis_set]
            outcome_true_vals = healthy_true_vals.loc[in_analysis_set,:]
            targ_true_vals = outcome_true_vals[targ]

            #validation
            val_outcome_preds = val_preds.loc[in_analysis_set_val]
            validation_outcome_true_vals = healthy_validation_true_vals.loc[in_analysis_set_val,:]
            valid_targ_true_vals = validation_outcome_true_vals[targ]

            #iter predictions to calculate SD
            outcome_preds_over_iters = preds_over_iters.loc[in_analysis_set]
            val_outcome_preds_over_iters = val_preds_over_iters.loc[in_analysis_set_val]

            #double check that all indices are the same
            assert (preds.index == true_vals.index).all()
            assert (val_preds.index == validation_true_vals.index).all()

            metab_data = cal_biobank_data[cal_metabolites]
            metab_data = metab_data.loc[preds.index]
            searchspace_input = metab_data[metab_data.columns.values[metab_data.isna().sum() == 0]].copy()
            searchspace_input = searchspace_input.loc[outcome_preds.index]

            #constructing list of demographic and metabolomic features to keep
            # NOTE: Currently only filters on metabolite columns, but extension
            # to other demographics would require different logic
            # e.g., {columns are in metabolites} | searchspace_input.columns.isin(demographics_features)
            in_analysis_set_features = (searchspace_input.columns.isin(cal_metabolites))

            # Determine common feature set between the test data and validation data
            searchspace_input_val = valid_metab[valid_metab.columns.values[valid_metab.isna().sum() == 0]].copy()
            searchspace_input_val = searchspace_input_val.loc[val_outcome_preds.index]
            in_analysis_set_val_features = (searchspace_input_val.columns.isin(cal_metabolites))

            searchspace_input = searchspace_input[searchspace_input.columns[in_analysis_set_features]]
            searchspace_input_val = searchspace_input_val[searchspace_input_val.columns[in_analysis_set_val_features]]
            common_cols = np.intersect1d(
                searchspace_input.columns, searchspace_input_val.columns)
            searchspace_input = searchspace_input[common_cols]
            searchspace_input_val = searchspace_input_val[common_cols]

            #compile list of features which need to be transformed into quantiles
            # NOTE: Categorical features need to be protected from quantile transformation
            to_transform =  searchspace_input.columns.isin(cal_metabolites) | searchspace_input.apply(is_numeric_dtype)
            searchspace_data = {}

            #transform all data into various quantiles [2,3,5] as per martins experiment
            for c in searchspace_input.columns:
                if c in to_transform:
                    for q in [2,3,5]:
                        column = f"{c}_q-{q}"
                        searchspace_data[column] = pd.qcut(
                            searchspace_input[c], q, duplicates="drop").cat.codes
                else:
                    searchspace_data[c] = searchspace_input[c]

            searchspace_data = pd.DataFrame(searchspace_data)

            is_metabolite = searchspace_data.columns.str.replace(
                r'_q.*$', '').isin(cal_metabolites)

            #compile list of features which need to be transformed into quantiles
            # NOTE: If you have categorical features - they need to be protected before.
            to_transform = searchspace_input_val.columns.isin(cal_metabolites) | searchspace_input_val.apply(is_numeric_dtype)

            searchspace_val_data = {}
            #transform all data into various quantiles [2,3,5] as per martins experiment
            for c in searchspace_input_val.columns:
                if c in to_transform:
                    for q in [2,3,5]:
                        column = f"{c}_q-{q}"
                        searchspace_val_data[column] = pd.qcut(
                            searchspace_input_val[c], q, duplicates="drop").cat.codes
                else:
                    searchspace_val_data[c] = searchspace_input_val[c]

            searchspace_val_data = pd.DataFrame(searchspace_val_data)
            is_val_metabolite = searchspace_val_data.columns.str.replace(
                r'_q.*$', '').isin(cal_metabolites)

            # INIT subgroup discovery objects and procedure
            target = ps.PredictionTarget(targ_true_vals.to_numpy(), outcome_preds.to_numpy(), evaluation_metric)
            searchspace = ps.create_selectors(searchspace_data[searchspace_data.columns[is_metabolite]])
            task = ps.SubgroupDiscoveryTask(
                searchspace_data,
                target,
                searchspace,
                result_set_size=subgroup_sizes[outcome],
                depth=4,
                qf=ps.PredictionQFNumeric(a=subgroup_alphas[outcome])
            )

            results = ps.BeamSearch(beam_width=subgroup_sizes[outcome]).execute(task)

            # Iterate over subgroups, collecting performance metrics
            subgroup_desc_df = results.to_dataframe()

            all_subgroups = results.results
            summary_stats = []
            total_merge_description = ''
            # Book-keeping for a continuously growing merging of subgroups
            total_merge_mask = np.full((len(searchspace_data.index)), False)
            for sg_num, sg in enumerate(all_subgroups):
                sg_quality, sg_description, qf = sg
                sg_mask, sg_size = ps.get_cover_array_and_size(sg_description, data=searchspace_data)

                # To be used to calculate statistics on a merged top 1:current_subgroup (as a running total)
                total_merge_mask = np.logical_or(total_merge_mask, sg_mask)

                # Iterating through subgroups to add custom stats (i.e., AUROC and AUPRC)
                runtotal_scorer = SubgroupScorer(
                    'Total Merged Subgroup', f'Merged from {sg_num} subgroups', total_merge_mask,
                    true_vals=targ_true_vals, preds=outcome_preds,
                    multiple_iter_preds=outcome_preds_over_iters)
                AUROC = runtotal_scorer.score_auroc()
                AUPRC = runtotal_scorer.score_auprc()
                AUROC_mean, AUROC_sd = runtotal_scorer.score_auroc(score_over_iters=True)
                AUPRC_mean, AUPRC_sd = runtotal_scorer.score_auprc(score_over_iters=True)
                # Next calculate the AUPRC in the subgroup
                subgroup_scorer = SubgroupScorer(
                    sg_num, sg_description, sg_mask,
                    true_vals=targ_true_vals, preds=outcome_preds,
                    multiple_iter_preds=outcome_preds_over_iters)
                subgroup_AUROC = subgroup_scorer.score_auroc()
                subgroup_AUPRC = subgroup_scorer.score_auprc()

                subgroup_AUROC_mean, subgroup_AUROC_sd = subgroup_scorer.score_auroc(
                    score_over_iters=True)
                subgroup_AUPRC_mean, subgroup_AUPRC_sd = subgroup_scorer.score_auprc(
                    score_over_iters=True)

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
                'No Subgroup Mask', 'Entire Dataset', np.full(len(outcome_preds), True),
                true_vals=targ_true_vals, preds=outcome_preds,
                multiple_iter_preds=outcome_preds_over_iters)
            kfold_AUROC = nomask_scorer.score_auroc()
            kfold_AUPRC = nomask_scorer.score_auprc()

            kfold_AUROC_mean, kfold_AUROC_sd = nomask_scorer.score_auroc(
                score_over_iters=True)
            kfold_AUPRC_mean, kfold_AUPRC_sd = nomask_scorer.score_auprc(
                score_over_iters=True)

            # NOTE: This applies the subgroup descriptions to the validation data
            all_subgroups = results.results
            summary_stats = []
            total_merge_description = ''
            total_merge_mask = np.full((len(searchspace_val_data.index)), False)

            for sg_num, sg in enumerate(all_subgroups):
                sg_quality, sg_description, qf = sg
                sg_mask, sg_size = ps.get_cover_array_and_size(sg_description, data=searchspace_val_data)

                # To be used to calculate statistics on a merged top 1:current_subgroup (as running total)
                total_merge_mask = np.logical_or(total_merge_mask, sg_mask)
                runtotal_scorer = SubgroupScorer(
                    'Total Merged Subgroup', f'Merged from {sg_num} subgroups', total_merge_mask,
                    true_vals=valid_targ_true_vals, preds=val_outcome_preds,
                    multiple_iter_preds=val_outcome_preds_over_iters)
                AUROC = runtotal_scorer.score_auroc()
                AUPRC = runtotal_scorer.score_auprc()
                AUROC_mean, AUROC_sd = runtotal_scorer.score_auroc(score_over_iters=True)
                AUPRC_mean, AUPRC_sd = runtotal_scorer.score_auprc(score_over_iters=True)

                # Next calculate the AUPRC in the subgroup
                subgroup_scorer = SubgroupScorer(
                    sg_num, sg_description, sg_mask,
                    true_vals=valid_targ_true_vals, preds=val_outcome_preds,
                    multiple_iter_preds=val_outcome_preds_over_iters)
                subgroup_AUROC = subgroup_scorer.score_auroc()
                subgroup_AUPRC = subgroup_scorer.score_auprc()
                subgroup_AUROC_mean, subgroup_AUROC_sd = subgroup_scorer.score_auroc(
                    score_over_iters=True)
                subgroup_AUPRC_mean, subgroup_AUPRC_sd = subgroup_scorer.score_auprc(
                    score_over_iters=True)

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

            subgroup_val_results_df = pd.DataFrame(summary_stats)

            #compile metrics of performance on validation set WITHOUT any subgroup masking
            nomask_scorer = SubgroupScorer(
                'No Subgroup Mask', 'Entire Dataset', np.full(len(val_outcome_preds), True),
                true_vals=valid_targ_true_vals, preds=val_outcome_preds,
                multiple_iter_preds=val_outcome_preds_over_iters)
            val_AUROC = nomask_scorer.score_auroc()
            val_AUPRC = nomask_scorer.score_auprc()

            val_AUROC_mean, val_AUROC_sd = nomask_scorer.score_auroc(
                score_over_iters=True)
            val_AUPRC_mean, val_AUPRC_sd = nomask_scorer.score_auprc(
                score_over_iters=True)

            #create random vectors on the test and validation data
            np.random.seed(1234)
            rand_pred = np.random.uniform(0,1,len(outcome_true_vals[targ]))
            rand_scorer = SubgroupScorer(
                'Random', 'Evaluation against random predictions',
                np.full(len(targ_true_vals), True),
                true_vals=targ_true_vals, preds=rand_pred)
            rand_AUROC = rand_scorer.score_auroc()
            rand_AUPRC = rand_scorer.score_auprc()

            rand_val_pred = np.random.uniform(0,1,len(validation_outcome_true_vals[targ]))
            rand_val_scorer = SubgroupScorer(
                'Random on Validation',
                'Evaluation against random predictions',
                np.full(len(valid_targ_true_vals), True),
                true_vals=valid_targ_true_vals, preds=rand_val_pred)
            rand_val_AUROC = rand_val_scorer.score_auroc()
            rand_val_AUPRC = rand_val_scorer.score_auprc()

            # Usually the top 20 percentile of data
            select = (subgroup_results_df[r"%data"] * 100)
            select_index = select.index[select == min(select, key=lambda x:abs(x-20))][0]

            # Iterate over subgroups, collecting performance metrics ONLY in the top
            # 20% subgroup
            subgroup_desc_df = results.to_dataframe()

            all_subgroups = results.results
            summary_stats = []
            total_merge_description = ''
            total_merge_mask = np.full((len(searchspace_data.index)), False)
            for sg_num, sg in enumerate(all_subgroups):
                sg_quality, sg_description, qf = sg
                sg_mask, sg_size = ps.get_cover_array_and_size(sg_description, data=searchspace_data)
                total_merge_mask = np.logical_or(total_merge_mask, sg_mask)

                if sg_num == select_index:
                    preds_top_subgroups = outcome_preds[total_merge_mask]
                    true_vals_top_subgroups = targ_true_vals[total_merge_mask]
                    random_preds_top_subgroups = rand_pred[total_merge_mask]
                    assert sorted(preds_top_subgroups.index) == sorted(true_vals_top_subgroups.index)
                    outcome_top_subgroups_df = pd.DataFrame.from_dict(
                        {'row_id': preds_top_subgroups.index,
                         'preds': preds_top_subgroups,
                         'true_vals': true_vals_top_subgroups,
                         'outcome': targ,
                         'evaluation_metric': metric,
                         'dataset': 'kfold_test'}).set_index('row_id')
                    top_k_subgroup_predictions.append(outcome_top_subgroups_df)

                    top20_scorer = SubgroupScorer(
                        'Top 20', 'Top 20 Percent of Data', total_merge_mask,
                        true_vals=targ_true_vals, preds=outcome_preds,
                        multiple_iter_preds=outcome_preds_over_iters)
                    kfold_AUROC_20 = top20_scorer.score_auroc()
                    ROC_tuple_20 = top20_scorer._score_overall_auroc(return_tuple=True)
                    kfold_AUPRC_20 = top20_scorer.score_auprc()
                    PR_tuple_20 = top20_scorer._score_overall_auprc(return_tuple=True)
                    kfold_AUROC_20_mean, kfold_AUROC_20_sd = top20_scorer.score_auroc(
                        score_over_iters=True)
                    kfold_AUPRC_20_mean, kfold_AUPRC_20_sd = top20_scorer.score_auprc(
                        score_over_iters=True)

                    top20_rand_scorer = SubgroupScorer(
                       'Top 20 Random', r'Random Predictions on top 20%data', total_merge_mask,
                        true_vals=targ_true_vals, preds=rand_pred)
                    rand_AUROC_20 = top20_rand_scorer.score_auroc()
                    rand_AUPRC_20 = top20_rand_scorer.score_auprc()

                    # Save predictions in top subgroups over iters
                    preds_iters_top_subgroups = (outcome_preds_over_iters[total_merge_mask]
                        .reset_index()
                        .melt(id_vars='row_id', value_name='preds')
                        .set_index('row_id'))
                    top_subgroups_iters_df = (pd.merge(
                        preds_iters_top_subgroups,
                        true_vals_top_subgroups.rename('true_vals'),
                        left_index=True, right_index=True))
                    top_subgroups_iters_df['outcome'] = targ
                    top_subgroups_iters_df['evaluation_metric'] = metric
                    top_subgroups_iters_df['dataset'] = 'kfold_test'
                    top_k_subgroup_preds_iters.append(top_subgroups_iters_df)
                    break

            # TODO: Try using the top 20% of data on the train data, don't worry about evaluating at each step
            #select = (subgroup_val_results_df[r"%data"] * 100)
            select = (subgroup_results_df[r'%data'] * 100)
            select_index = select.index[select == min(select, key=lambda x:abs(x-20))][0]
            total_merge_mask = np.full((len(searchspace_val_data.index)), False)

            # This gets the performance for the top 20% of subgroups in the validation dataset
            for sg_num, sg in enumerate(all_subgroups):
                sg_quality, sg_description, qf = sg
                sg_mask, sg_size = ps.get_cover_array_and_size(sg_description, data=searchspace_val_data)
                total_merge_mask = np.logical_or(total_merge_mask, sg_mask)
                if sg_num == select_index:
                    preds_top_subgroups = val_outcome_preds[total_merge_mask]
                    true_vals_top_subgroups = valid_targ_true_vals[total_merge_mask]
                    random_preds_top_subgroups = rand_val_pred[total_merge_mask]
                    assert sorted(preds_top_subgroups.index) == sorted(true_vals_top_subgroups.index)
                    outcome_top_subgroups_df = pd.DataFrame.from_dict(
                        {val_index_col: preds_top_subgroups.index,
                         'preds': preds_top_subgroups,
                         'true_vals': true_vals_top_subgroups,
                         'outcome': targ,
                         'evaluation_metric': metric,
                         'dataset': 'holdout_validation'}).set_index(val_index_col)
                    top_k_subgroup_predictions.append(outcome_top_subgroups_df)

                    top20_scorer = SubgroupScorer(
                        'Top 20', 'Top 20 Percent of Data', total_merge_mask,
                        true_vals=valid_targ_true_vals, preds=val_outcome_preds,
                        multiple_iter_preds=val_outcome_preds_over_iters)
                    val_AUROC_20 = top20_scorer.score_auroc()
                    ROC_tuple_20_val = top20_scorer._score_overall_auroc(return_tuple=True)
                    val_AUPRC_20 = top20_scorer.score_auprc()
                    PR_tuple_20_val = top20_scorer._score_overall_auprc(return_tuple=True)
                    val_AUROC_20_mean, val_AUROC_20_sd = top20_scorer.score_auroc(
                        score_over_iters=True)
                    val_AUPRC_20_mean, val_AUPRC_20_sd = top20_scorer.score_auprc(
                        score_over_iters=True)

                    top20_rand_scorer = SubgroupScorer(
                       'Top 20 Random', r'Random Predictions on top 20%data', total_merge_mask,
                        true_vals=valid_targ_true_vals, preds=rand_val_pred)
                    rand_val_AUROC_20 = top20_rand_scorer.score_auroc()
                    rand_val_AUPRC_20 = top20_rand_scorer.score_auprc()

                    # Save results over iters
                    preds_iters_top_subgroups = (val_outcome_preds_over_iters[total_merge_mask]
                        .reset_index()
                        .melt(id_vars=val_index_col, value_name='preds')
                        .set_index(val_index_col))
                    top_subgroups_iters_df = (pd.merge(
                        preds_iters_top_subgroups,
                        true_vals_top_subgroups.rename('true_vals'),
                        left_index=True, right_index=True))
                    top_subgroups_iters_df['outcome'] = targ
                    top_subgroups_iters_df['evaluation_metric'] = metric
                    top_subgroups_iters_df['dataset'] = 'external_validation'
                    top_k_subgroup_preds_iters.append(top_subgroups_iters_df)
                    break

            iter_results[targ+col_annotation] = [kfold_AUROC_mean, kfold_AUROC_sd, val_AUROC_mean, val_AUROC_sd, kfold_AUROC_20_mean, kfold_AUROC_20_sd, val_AUROC_20_mean, val_AUROC_20_sd,
            kfold_AUPRC_mean, kfold_AUPRC_sd, val_AUPRC_mean, val_AUPRC_sd, kfold_AUPRC_20_mean, kfold_AUPRC_20_sd, val_AUPRC_20_mean, val_AUPRC_20_sd]
            all_results[targ+col_annotation] = [subgroup_results_df, subgroup_val_results_df, kfold_AUROC, kfold_AUPRC, val_AUROC, val_AUPRC, PR_tuple_20, PR_tuple_20_val, kfold_AUPRC_20, val_AUPRC_20, ROC_tuple_20, ROC_tuple_20_val, kfold_AUROC_20, val_AUROC_20, rand_AUROC, rand_AUPRC, rand_val_AUROC, rand_val_AUPRC, rand_AUROC_20, rand_AUPRC_20, rand_val_AUROC_20, rand_val_AUPRC_20]

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
            targ=outcome+"_any"
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
            (precision_val_20, recall_val_20, thresholds_val_20) = all_results[targ+col_annotation][7]

            kfold_20_AUPRC = all_results[targ+col_annotation][8]
            val_20_AUPRC = all_results[targ+col_annotation][9]

            (fpr_20, tpr_20, thresholds_20) = all_results[targ+col_annotation][10]
            (fpr_val_20, tpr_val_20, thresholds_val_20) = all_results[targ+col_annotation][11]

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
            worksheet_20.write(0,1, "kfold AUPRC")
            worksheet_20.write(0,2, "val AUPRC")
            worksheet_20.write(0,3, "kfold AUROC")
            worksheet_20.write(0,4, "val AUROC")

            worksheet_20.write(outcome_order.index(outcome)+1, 0, outcome)
            worksheet_20.write(outcome_order.index(outcome)+1, 1, kfold_20_AUPRC)
            worksheet_20.write(outcome_order.index(outcome)+1, 2, val_20_AUPRC)
            worksheet_20.write(outcome_order.index(outcome)+1, 3, kfold_20_AUROC)
            worksheet_20.write(outcome_order.index(outcome)+1, 4, val_20_AUROC)


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
            worksheet_20_mean.write(0,1, "kfold AUPRC")
            worksheet_20_mean.write(0,2, "val AUPRC")
            worksheet_20_mean.write(0,3, "kfold AUROC")
            worksheet_20_mean.write(0,4, "val AUROC")

            worksheet_20_mean.write(0,6, "kfold AUPRC SD")
            worksheet_20_mean.write(0,7, "val AUPRC SD")
            worksheet_20_mean.write(0,8, "kfold AUROC SD")
            worksheet_20_mean.write(0,9, "val AUROC SD")


            worksheet_20_mean.write(outcome_order.index(outcome)+1, 0, outcome)
            worksheet_20_mean.write(outcome_order.index(outcome)+1, 1, kfold_AUPRC_mean_20)
            worksheet_20_mean.write(outcome_order.index(outcome)+1, 2, val_AUPRC_mean_20)
            worksheet_20_mean.write(outcome_order.index(outcome)+1, 3, kfold_AUROC_mean_20)
            worksheet_20_mean.write(outcome_order.index(outcome)+1, 4, val_AUROC_mean_20)

            worksheet_20_mean.write(outcome_order.index(outcome)+1, 6, kfold_AUPRC_sd_20)
            worksheet_20_mean.write(outcome_order.index(outcome)+1, 7, val_AUPRC_sd_20)
            worksheet_20_mean.write(outcome_order.index(outcome)+1, 8, kfold_AUROC_sd_20)
            worksheet_20_mean.write(outcome_order.index(outcome)+1, 9, val_AUROC_sd_20)

            worksheet_baseline_mean.write(0,0, "outcome")
            worksheet_baseline_mean.write(0,1, "kfold AUROC")
            worksheet_baseline_mean.write(0,2, "kfold AUPRC")
            worksheet_baseline_mean.write(0,3, "val AUROC")
            worksheet_baseline_mean.write(0,4, "val AUPRC")

            worksheet_baseline_mean.write(0,6, "kfold AUPRC SD")
            worksheet_baseline_mean.write(0,7, "val AUPRC SD")
            worksheet_baseline_mean.write(0,8, "kfold AUROC SD")
            worksheet_baseline_mean.write(0,9, "val AUROC SD")

            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 0, outcome)
            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 1, kfold_AUROC_mean)
            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 2, kfold_AUPRC_mean)
            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 3, val_AUROC_mean)
            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 4, val_AUPRC_mean)

            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 6, kfold_AUPRC_sd)
            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 7, val_AUPRC_sd)
            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 8, kfold_AUROC_sd)
            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 9, val_AUROC_sd)


            rand_AUROC = all_results[targ+col_annotation][14]
            rand_AUPRC = all_results[targ+col_annotation][15]

            rand_val_AUROC = all_results[targ+col_annotation][16]
            rand_val_AUPRC = all_results[targ+col_annotation][17]


            rand_AUROC_20 = all_results[targ+col_annotation][18]
            rand_AUPRC_20 = all_results[targ+col_annotation][19]

            rand_val_AUROC_20 = all_results[targ+col_annotation][20]
            rand_val_AUPRC_20 = all_results[targ+col_annotation][21]

            worksheet_20_rand.write(0,0, "outcome")
            worksheet_20_rand.write(0,1, "kfold AUPRC")
            worksheet_20_rand.write(0,2, "val AUPRC")
            worksheet_20_rand.write(0,3, "kfold AUROC")
            worksheet_20_rand.write(0,4, "val AUROC")

            worksheet_20_rand.write(outcome_order.index(outcome)+1, 0, outcome)
            worksheet_20_rand.write(outcome_order.index(outcome)+1, 1, rand_AUPRC_20)
            worksheet_20_rand.write(outcome_order.index(outcome)+1, 2, rand_val_AUPRC_20)
            worksheet_20_rand.write(outcome_order.index(outcome)+1, 3, rand_AUROC_20)
            worksheet_20_rand.write(outcome_order.index(outcome)+1, 4, rand_val_AUROC_20)

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

            #adding column titles
            col_num=0
            row_num=0
            for col in subgroup_val_results.columns:
                temp = worksheet_val.write(row_num, col_num, col)
                col_num = col_num + 1

            #adding subgroup result vectors
            col_num=0
            for col in subgroup_results.columns:
                row_num = 1
                for val in subgroup_val_results[col]:
                    if pd.isna(val):
                        temp = worksheet_val.write(row_num, col_num, "nan")
                    else:
                        temp = worksheet_val.write(row_num, col_num, val)
                    row_num  = row_num + 1
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
            if not pd.isna(precision_val_20).all() and not pd.isna(recall_val_20).all():
                for row_num, (prec, rec) in enumerate(zip(precision_val_20, recall_val_20)):
                    worksheet_pr_val.write(row_num+1,0,prec)
                    worksheet_pr_val.write(row_num+1,1,rec)

            # adding ROC tpr and fpr axes
            worksheet_roc.write(0,0, "TPR KFold")
            worksheet_roc.write(0,1, "FPR KFold")
            worksheet_roc_val.write(0,0, "TPR Val")
            worksheet_roc_val.write(0,1, "FPR Val")
            if not pd.isna(tpr_20).all() and not pd.isna(fpr_20).all():
                for row_num, (tpr, fpr) in enumerate(zip(tpr_20, fpr_20)):
                    worksheet_roc.write(row_num+1,0,tpr)
                    worksheet_roc.write(row_num+1,1,fpr)
            if not pd.isna(tpr_val_20).all() and not pd.isna(fpr_val_20).all():
                for row_num, (tpr, fpr) in enumerate(zip(tpr_val_20, fpr_val_20)):
                    worksheet_roc_val.write(row_num+1,0,tpr)
                    worksheet_roc_val.write(row_num+1,1,fpr)

        workbook.close()
        logger.info(f'Writing .csv to {output_dir}')

    # Save top K subgroup predictions
    top_k_subgroup_predictions = pd.concat(top_k_subgroup_predictions)
    top_k_subgroup_predictions.to_csv(
        output_dir + 'top_k_subgroup_predictions.csv')
    top_k_subgroup_preds_iters = pd.concat(top_k_subgroup_preds_iters)
    top_k_subgroup_preds_iters.to_csv(
        output_dir + 'top_k_subgroup_preds_over_iters.csv')


if __name__ == '__main__':
    args = get_args()
    main(args)
