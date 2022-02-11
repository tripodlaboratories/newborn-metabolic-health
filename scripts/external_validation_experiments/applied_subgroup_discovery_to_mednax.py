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

    # TODO: Debugging Numpy Errors
    np.seterr(all='raise')

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
    preds = preds.groupby(["row_id"])["bottleneck_unit_0"].mean() #smart way (it is, double checked against stupid for loop method)

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
        logger.info(f'starting subgroup discovery with metric: {metric}')

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
            in_analysis_set = (
                true_vals[other_outcomes].sum(axis=1) == 0) | (true_vals[targ] == 1)
            in_analysis_set_val = (
                validation_true_vals[other_outcomes].sum(axis=1) == 0) | (validation_true_vals[targ] == 1)

            # NOTE: Since we are interested in identifying HEALTHY individuals
            # The target for prediction will be switched to healthy obs
            true_vals = 1 - true_vals # this along with a change in the 'in_analysis_set' vector is the only change
            validation_true_vals = 1 - validation_true_vals[outcome_labels]

            #k-fold
            outcome_preds = preds.loc[in_analysis_set]
            outcome_true_vals = true_vals.loc[in_analysis_set,:]

            #validation
            val_outcome_preds = val_preds.loc[in_analysis_set_val]
            validation_outcome_true_vals = validation_true_vals.loc[in_analysis_set_val,:]

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
            #temp_data = outcome_subset_data[outcome_subset_data.columns.values[outcome_subset_data.isna().sum() == 0]].copy()

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
            target = ps.PredictionTarget(outcome_true_vals[targ].to_numpy(), outcome_preds.to_numpy(), evaluation_metric)
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
            # TODO: Extract the subgroup discovery logic and apply to the Mednax dataset
            # TODO: Remove commented examples from the final script
            # EXAMPLES
            # pickle after removing data, if you had to.
            # results.task.data = None
            # pickle.dump(...)
            # results to dataframe (hard to add custom statistics)
            results.to_dataframe()
            results.results # all the subgroups
            # get information from THE FIRST subgroup
            sg_quality, sg_description, qf = results.results[0]
            # get subgroup mask / selection array for ONE subgroup
            sg_msk, sg_size = ps.get_cover_array_and_size(sg_description, data=searchspace_data)

            # TODO: Refactor the subgroup iteration
            subgroup_desc_df = results.to_dataframe()
            all_subgroups = results.results
            # NOTE: This is where the original logic resumes.
            subgroup_desc = results.to_dataframe()["subgroup"]

            # TODO: data should be called something more like summary_stats, or overall_summary_stats
            summary_stats = []
            total_merge_description = ""
            # TODO: Refactoring the Subgroup Discovery Iteration
            # TODO: Need to add in logic for iteratively growing the subgroup slice.
            # TODO: Which is probably what the elementwise_or thing is doing.
            # This replaces bool_vec
            # Book-keeping for a continuously growing merging of subgroups
            subgroup_merge_mask = np.full((len(searchspace_data.index)), False)
            for sg_num, sg in enumerate(all_subgroups):
                sg_quality, sg_description, qf = sg
                sg_msk, sg_size = ps.get_cover_array_and_size(sg_description, data=searchspace_data)

                # To be used to calculate statistics on a merged top 1:current_subgroup
                subgroup_merge_mask = np.logical_or(subgroup_merge_mask, sg_msk)

                # NOTE: The next logic exists because it's difficult to add custom stats to the dataframe original output
                precision, recall, thresholds = precision_recall_curve(
                    outcome_true_vals[targ][subgroup_merge_mask], outcome_preds[subgroup_merge_mask])
                AUPRC = auc(recall, precision)
                precision, recall, thresholds = precision_recall_curve(
                    outcome_true_vals[targ][sg_msk], outcome_preds[sg_msk])
                subgroup_AUPRC = auc(recall, precision)
                #append all data to array
                #compile metrics of performancpe
                n_prediction_iters = len(outcome_preds_over_iters.columns)
                # TODO: Clarify what this condition is checking for...?
                if(len(outcome_true_vals[targ][subgroup_merge_mask]) == outcome_true_vals[targ][subgroup_merge_mask].sum()):
                    AUROC = np.nan
                    AUROC_sd = np.nan
                    AUROC_mean = np.nan

                    # Loops like this collect performance of individual model iterations
                    temp_auprc = []
                    for i in range(n_prediction_iters):
                        precision, recall, thresholds = precision_recall_curve(
                            outcome_true_vals[targ][sg_msk],
                            outcome_preds_over_iters.iloc[:,i][sg_msk])
                        temp_auprc.append(auc(recall, precision))

                    AUPRC_sd = np.std(temp_auprc, ddof=1) #used to make std consistent with R
                    AUPRC_mean = np.mean(temp_auprc)

                else:
                    AUROC = roc_auc_score(
                        outcome_true_vals[targ][subgroup_merge_mask],
                        outcome_preds[subgroup_merge_mask])

                    temp_auroc = []
                    temp_auprc = []
                    for i in range(n_prediction_iters):
                        precision, recall, thresholds = precision_recall_curve(
                            outcome_true_vals[targ][subgroup_merge_mask],
                            outcome_preds_over_iters.iloc[:,i][subgroup_merge_mask])
                        temp_auprc.append(auc(recall, precision))
                        temp_auroc.append(roc_auc_score(
                            outcome_true_vals[targ][subgroup_merge_mask],
                            outcome_preds_over_iters.iloc[:,i][subgroup_merge_mask]))

                    # Handling cases where validation data is not made over repeated iterations
                    if len(temp_auroc) == 1 or len(temp_auprc) == 1:
                        AUROC_sd = 0
                        AUPRC_sd = 0
                    else:
                        AUROC_sd = np.std(temp_auroc, ddof=1)
                        AUPRC_sd = np.std(temp_auprc, ddof=1)
                    AUROC_mean = np.mean(temp_auroc)
                    AUPRC_mean = np.mean(temp_auprc)

                if(len(outcome_true_vals[targ][subgroup_msk]) == outcome_true_vals[targ][subgroup_msk].sum()):
                    subgroup_AUROC = np.nan
                    subgroup_AUROC_sd = np.nan
                    subgroup_AUROC_mean = np.nan

                    temp_auprc = []
                    # Collects statistics across each iteration. Each column represented one iteration of repeated K-Fold
                    for i in range(n_prediction_iters):
                        precision, recall, thresholds = precision_recall_curve(
                            outcome_true_vals[targ][subgroup_msk],
                            outcome_preds_over_iters.iloc[:,i][subgroup_msk])
                        temp_auprc.append(auc(recall, precision))

                    subgroup_AUPRC_sd = np.std(temp_auprc, ddof=1)
                    subgroup_AUPRC_mean = np.mean(temp_auprc)

                else:
                    subgroup_AUROC = roc_auc_score(
                        outcome_true_vals[targ][subgroup_msk],
                        outcome_preds[subgroup_msk])

                    temp_auroc = []
                    temp_auprc = []
                    for i in range(n_prediction_iters):
                        precision, recall, thresholds = precision_recall_curve(
                            outcome_true_vals[targ][subgroup_msk],
                            outcome_preds_over_iters.iloc[:,i][subgroup_msk])
                        temp_auprc.append(auc(recall, precision))
                        temp_auroc.append(roc_auc_score(
                            outcome_true_vals[targ][subgroup_msk],
                            outcome_preds_over_iters.iloc[:,i][subgroup_msk]))

                    subgroup_AUROC_sd = np.std(temp_auroc, ddof=1)
                    subgroup_AUROC_mean = np.mean(temp_auroc)
                    subgroup_AUPRC_sd = np.std(temp_auprc, ddof=1)
                    subgroup_AUPRC_mean = np.mean(temp_auprc)

                total_merge_description = subgroup_description + "-OR-" + total_merge_description

                summary_stats.append({
                    'total_subgroup_merge': total_merge_description,
                    'subgroup': subgroup_description,
                    'size': subgroup_merge_mask.sum(),
                    r'%data': subgroup_size / len(searchspace_data.index),
                    "subgroup size": subgroup_size,
                    "subgroup number": subgroup_num,
                    "AUPRC": AUPRC,
                    "subgroup AUPRC": subgroup_AUPRC,
                    "AUROC": AUROC,
                    "subgroup AUROC": subgroup_AUROC})

            # Gather results from all subgroups
            subgroup_results_df = pd.DataFrame(summary_stats)

            #compile metrics of performancpe
            kfold_AUROC = roc_auc_score(outcome_true_vals[targ], outcome_preds)
            precision, recall, thresholds = precision_recall_curve(outcome_true_vals[targ], outcome_preds)
            kfold_AUPRC = auc(recall, precision)

            temp_auroc = []
            temp_auprc = []
            for i in range(n_prediction_iters):
                precision, recall, thresholds = precision_recall_curve(outcome_true_vals[targ], outcome_preds_over_iters.iloc[:,i])
                temp_auprc.append(auc(recall, precision))
                temp_auroc.append(roc_auc_score(outcome_true_vals[targ], outcome_preds_over_iters.iloc[:,i]))

            kfold_AUROC_sd = np.std(temp_auroc, ddof=1)
            kfold_AUROC_mean = np.mean(temp_auroc)
            kfold_AUPRC_sd = np.std(temp_auprc, ddof=1)
            kfold_AUPRC_mean = np.mean(temp_auprc)

            # TODO: Clarify the logic here: This applies the subgroup descriptions to the validation data
            data = []
            total_elem = ""
            bool_vec = np.full((len(searchspace_val_data.index)), False)
            count = 0
            for elem in subgroup_desc:
                count = count + 1
                bool_vec_inner = np.full((len(searchspace_val_data.index)), True)
                for cond in elem.split(" AND "):
                    if("==" in cond):
                        splt = cond.split("==")
                        bool_vec_inner = bool_vec_inner & (searchspace_val_data[splt[0]] == (int(splt[1]) if "'" not in splt[1] and "." not in splt[1] else float(splt[1]) if "." in splt[1] else splt[1].replace("'","") ))
                    if(">=" in cond):
                        splt = cond.split(">=")
                        bool_vec_inner = bool_vec_inner & (searchspace_val_data[splt[0]] >= (int(splt[1]) if "'" not in splt[1] and "." not in splt[1] else float(splt[1]) if "." in splt[1] else splt[1].replace("'","") ))
                    if("<=" in cond):
                        splt = cond.split("<=")
                        bool_vec_inner = bool_vec_inner & (searchspace_val_data[splt[0]] <= (int(splt[1]) if "'" not in splt[1] and "." not in splt[1] else float(splt[1]) if "." in splt[1] else splt[1].replace("'","") ))
                bool_vec = np.logical_or(bool_vec,  bool_vec_inner)

                precision, recall, thresholds = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec], val_outcome_preds[bool_vec])
                AUPRC = auc(recall, precision)

                # TODO: Debugging for the case where a subgroup is NOT FOUND in the validation dataset
                try:
                    precision, recall, thresholds = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec_inner], val_outcome_preds[bool_vec_inner])
                    subgroup_AUPRC = auc(recall, precision)
                except ValueError as e:
                    logger.warn(f'No individuals found in subgroup {count-1}, {subgroup_desc}')
                    subgroup_AUPRC = np.nan

                #precision, recall, thresholds = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec_inner], val_outcome_preds[bool_vec_inner])
                #subgroup_AUPRC = auc(recall, precision)

                n_validation_prediction_iters = len(val_outcome_preds_over_iters.columns)
                if(len(validation_outcome_true_vals[targ][bool_vec]) == validation_outcome_true_vals[targ][bool_vec].sum()):
                    AUROC = np.nan
                    AUROC_sd = np.nan
                    AUROC_mean = np.nan

                    temp_auprc = []
                    for i in range(n_validation_prediction_iters):
                        precision, recall, thresholds = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec], val_outcome_preds_over_iters.iloc[:,i][bool_vec])
                        temp_auprc.append(auc(recall, precision))

                    AUPRC_sd = np.std(temp_auprc, ddof=1)
                    AUPRC_mean = np.mean(temp_auprc)

                else:
                    AUROC = roc_auc_score(validation_outcome_true_vals[targ][bool_vec], val_outcome_preds[bool_vec])

                    temp_auroc = []
                    temp_auprc = []

                    for i in range(n_validation_prediction_iters):
                        precision, recall, thresholds = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec], val_outcome_preds_over_iters.iloc[:,i][bool_vec])
                        temp_auprc.append(auc(recall, precision))
                        temp_auroc.append(roc_auc_score(validation_outcome_true_vals[targ][bool_vec], val_outcome_preds_over_iters.iloc[:,i][bool_vec]))

                    if len(temp_auroc) == 1 or len(temp_auprc) == 1:
                        AUROC_sd = 0
                        AUPRC_sd = 0
                    else:
                        AUROC_sd = np.std(temp_auroc, ddof=1)
                        AUPRC_sd = np.std(temp_auprc, ddof=1)
                    AUROC_mean = np.mean(temp_auroc)
                    AUPRC_mean = np.mean(temp_auprc)

                if(len(validation_outcome_true_vals[targ][bool_vec_inner]) == validation_outcome_true_vals[targ][bool_vec_inner].sum()):
                    subgroup_AUROC = np.nan
                    subgroup_AUROC_sd = np.nan
                    subgroup_AUROC_mean = np.nan

                    temp_auprc = []
                    for i in range(n_validation_prediction_iters):
                        precision, recall, thresholds = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec_inner], val_outcome_preds_over_iters.iloc[:,i][bool_vec_inner])
                        temp_auprc.append(auc(recall, precision))

                    if len(temp_auprc) == 1:
                        subgroup_AUPRC_sd = 0
                    else:
                        subgroup_AUPRC_sd = np.std(temp_auprc, ddof=1)
                    subgroup_AUPRC_mean = np.mean(temp_auprc)

                else:
                    subgroup_AUROC = roc_auc_score(validation_outcome_true_vals[targ][bool_vec_inner], val_outcome_preds[bool_vec_inner])

                    #calculate sd metrics
                    temp_auroc = []
                    temp_auprc = []
                    for i in range(n_validation_prediction_iters):
                        precision, recall, thresholds = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec_inner], val_outcome_preds_over_iters.iloc[:,i][bool_vec_inner])
                        temp_auprc.append(auc(recall, precision))
                        temp_auroc.append(roc_auc_score(validation_outcome_true_vals[targ][bool_vec_inner], val_outcome_preds_over_iters.iloc[:,i][bool_vec_inner]))

                    if len(temp_auroc) == 1 or len(temp_auprc) == 1:
                        subgroup_AUROC_sd = 0
                        subgroup_AUPRC_sd = 0
                    else:
                        subgroup_AUROC_sd = np.std(temp_auroc, ddof=1)
                        subgroup_AUPRC_sd = np.std(temp_auprc, ddof=1)
                    subgroup_AUROC_mean = np.mean(temp_auroc)
                    subgroup_AUPRC_mean = np.mean(temp_auprc)


                total_elem = elem + "-OR-" + total_elem
                data.append([total_elem, elem, bool_vec.sum(), bool_vec.sum()/len(searchspace_val_data.iloc[:,0]), bool_vec_inner.sum(), count, AUPRC, subgroup_AUPRC, AUROC, subgroup_AUROC])

            subgroup_val_results_df = pd.DataFrame(data, columns=["total group","subgroup","size", "% data", "subgroup size", "num_groups", "AUPRC", "subgroup AUPRC", "AUROC", "subgroup AUROC"])

            val_AUROC = roc_auc_score(validation_outcome_true_vals[targ], val_outcome_preds)
            precision, recall, thresholds = precision_recall_curve(validation_outcome_true_vals[targ], val_outcome_preds)
            val_AUPRC = auc(recall, precision)

            temp_auroc = []
            temp_auprc = []
            for i in range(n_validation_prediction_iters):
                precision1, recall1, thresholds = precision_recall_curve(validation_outcome_true_vals[targ], val_outcome_preds_over_iters.iloc[:,i])
                temp_auprc.append(auc(recall1, precision1))
                temp_auroc.append(roc_auc_score(validation_outcome_true_vals[targ], val_outcome_preds_over_iters.iloc[:,i]))

            if len(temp_auroc) == 1 or len(temp_auprc) == 1:
                val_AUROC_sd = 0
                val_AUPRC_sd = 0
            else:
                val_AUROC_sd = np.std(temp_auroc, ddof=1)
                val_AUPRC_sd = np.std(temp_auprc, ddof=1)
            val_AUROC_mean = np.mean(temp_auroc)
            val_AUPRC_mean = np.mean(temp_auprc)

            #create random vector
            np.random.seed(1234)
            rand_val_pred = np.random.uniform(0,1,len(validation_outcome_true_vals[targ]))
            rand_pred = np.random.uniform(0,1,len(outcome_true_vals[targ]))

            rand_AUROC = roc_auc_score(outcome_true_vals[targ], rand_pred)
            rand_precision, rand_recall, rand_thresholds = precision_recall_curve(outcome_true_vals[targ], rand_pred)
            rand_AUPRC = auc(rand_recall, rand_precision)

            rand_val_AUROC = roc_auc_score(validation_outcome_true_vals[targ], rand_val_pred)
            rand_precision, rand_recall, rand_thresholds = precision_recall_curve(validation_outcome_true_vals[targ], rand_val_pred)
            rand_val_AUPRC = auc(rand_recall, rand_precision)

            # Usually the top 20 percentile of data
            select =  (subgroup_results_df["% data"]* 100)
            select_index = select.index[select == min(select, key=lambda x:abs(x-20))][0]
            bool_vec = np.full((len(searchspace_data.index)), False)

            # TODO: Maybe use direct slicing of the subgroups, and iterate over that?
            # FIXME: Replace string splitting with iteration over the subgroups and the subgroup mask
            # FIXME: Also, why is there a third for loop over the subgroup descriptions here?
            count = 0
            for elem in subgroup_desc:
                count = count + 1
                bool_vec_inner = np.full((len(searchspace_data.index)), True)
                for cond in elem.split(" AND "):
                    if("==" in cond):
                        splt = cond.split("==")
                        bool_vec_inner = bool_vec_inner & (searchspace_data[splt[0]] == (int(splt[1]) if "'" not in splt[1] and "." not in splt[1] else float(splt[1]) if "." in splt[1] else splt[1].replace("'","") ))
                    if(">=" in cond):
                        splt = cond.split(">=")
                        bool_vec_inner = bool_vec_inner & (searchspace_data[splt[0]] >= (int(splt[1]) if "'" not in splt[1] and "." not in splt[1] else float(splt[1]) if "." in splt[1] else splt[1].replace("'","") ))
                    if("<=" in cond):
                        splt = cond.split("<=")
                        bool_vec_inner = bool_vec_inner & (searchspace_data[splt[0]] <= (int(splt[1]) if "'" not in splt[1] and "." not in splt[1] else float(splt[1]) if "." in splt[1] else splt[1].replace("'","") ))
                bool_vec = np.logical_or(bool_vec,  bool_vec_inner)
                if count == select_index:
                    preds_top_subgroups = outcome_preds[bool_vec]
                    true_vals_top_subgroups = outcome_true_vals[targ][bool_vec]
                    random_preds_top_subgroups = rand_pred[bool_vec]
                    assert sorted(preds_top_subgroups.index) == sorted(true_vals_top_subgroups.index)
                    outcome_top_subgroups_df = pd.DataFrame.from_dict(
                        {'row_id': preds_top_subgroups.index,
                         'preds': preds_top_subgroups,
                         'true_vals': true_vals_top_subgroups,
                         'outcome': targ,
                         'evaluation_metric': metric,
                         'dataset': 'kfold_test'}).set_index('row_id')
                    top_k_subgroup_predictions.append(outcome_top_subgroups_df)

                    ROC_tuple_20 = roc_curve(outcome_true_vals[targ][bool_vec], outcome_preds[bool_vec])
                    kfold_AUROC_20 = auc(ROC_tuple_20[0], ROC_tuple_20[1])
                    PR_tuple_20 = precision_recall_curve(outcome_true_vals[targ][bool_vec], outcome_preds[bool_vec])
                    kfold_AUPRC_20 = auc(PR_tuple_20[1], PR_tuple_20[0])

                    rand_AUROC_20 = roc_auc_score(outcome_true_vals[targ][bool_vec], rand_pred[bool_vec])
                    rand_precision, rand_recall, rand_thresholds = precision_recall_curve(outcome_true_vals[targ][bool_vec], rand_pred[bool_vec])
                    rand_AUPRC_20 = auc(rand_recall, rand_precision)

                    temp_auroc = []
                    temp_auprc = []
                    for i in range(n_prediction_iters):
                        precision, recall, thresholds = precision_recall_curve(outcome_true_vals[targ][bool_vec], outcome_preds_over_iters.iloc[:,i][bool_vec])
                        temp_auprc.append(auc(recall, precision))
                        temp_auroc.append(roc_auc_score(outcome_true_vals[targ][bool_vec], outcome_preds_over_iters.iloc[:,i][bool_vec]))

                    kfold_AUROC_20_sd = np.std(temp_auroc, ddof=1)
                    kfold_AUROC_20_mean = np.mean(temp_auroc)
                    kfold_AUPRC_20_sd = np.std(temp_auprc, ddof=1)
                    kfold_AUPRC_20_mean = np.mean(temp_auprc)

                    # Save predictions in top subgroups over iters
                    preds_iters_top_subgroups = (outcome_preds_over_iters[bool_vec]
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

            select =  (subgroup_val_results_df["% data"]* 100)
            select_index = select.index[select == min(select, key=lambda x:abs(x-20))][0]
            bool_vec = np.full((len(searchspace_val_data.index)), False)

            # TODO: Clarify that this gets the performance for the top 20% of subgroups in the validation dataset
            # FIXME: Refactor this into iterating over the subgroups
            count = 0
            for elem in subgroup_desc:
                count = count + 1
                bool_vec_inner = np.full((len(searchspace_val_data.index)), True)
                for cond in elem.split(" AND "):
                    if("==" in cond):
                        splt = cond.split("==")
                        bool_vec_inner = bool_vec_inner & (searchspace_val_data[splt[0]] == (int(splt[1]) if "'" not in splt[1] and "." not in splt[1] else float(splt[1]) if "." in splt[1] else splt[1].replace("'","") ))
                    if(">=" in cond):
                        splt = cond.split(">=")
                        bool_vec_inner = bool_vec_inner & (searchspace_val_data[splt[0]] >= (int(splt[1]) if "'" not in splt[1] and "." not in splt[1] else float(splt[1]) if "." in splt[1] else splt[1].replace("'","") ))
                    if("<=" in cond):
                        splt = cond.split("<=")
                        bool_vec_inner = bool_vec_inner & (searchspace_val_data[splt[0]] <= (int(splt[1]) if "'" not in splt[1] and "." not in splt[1] else float(splt[1]) if "." in splt[1] else splt[1].replace("'","") ))
                bool_vec = np.logical_or(bool_vec,  bool_vec_inner)
                if count == select_index:
                    preds_top_subgroups = val_outcome_preds[bool_vec]
                    true_vals_top_subgroups = validation_outcome_true_vals[targ][bool_vec]
                    random_preds_top_subgroups = rand_val_pred[bool_vec]
                    assert sorted(preds_top_subgroups.index) == sorted(true_vals_top_subgroups.index)
                    outcome_top_subgroups_df = pd.DataFrame.from_dict(
                        {'row_id': preds_top_subgroups.index,
                         'preds': preds_top_subgroups,
                         'true_vals': true_vals_top_subgroups,
                         'outcome': targ,
                         'evaluation_metric': metric,
                         'dataset': 'holdout_validation'}).set_index('row_id')
                    top_k_subgroup_predictions.append(outcome_top_subgroups_df)

                    ROC_tuple_20_val = roc_curve(validation_outcome_true_vals[targ][bool_vec], val_outcome_preds[bool_vec])
                    val_AUROC_20 = auc(ROC_tuple_20_val[0], ROC_tuple_20_val[1])
                    PR_tuple_20_val = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec], val_outcome_preds[bool_vec])
                    val_AUPRC_20 = auc(PR_tuple_20_val[1], PR_tuple_20_val[0])

                    rand_val_AUROC_20 = roc_auc_score(validation_outcome_true_vals[targ][bool_vec], rand_val_pred[bool_vec])
                    rand_precision, rand_recall, rand_thresholds = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec], rand_val_pred[bool_vec])
                    rand_val_AUPRC_20 = auc(rand_recall, rand_precision)


                    temp_auroc = []
                    temp_auprc = []
                    for i in range(n_validation_prediction_iters):
                        precision, recall, thresholds = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec], val_outcome_preds_over_iters.iloc[:,i][bool_vec])
                        temp_auprc.append(auc(recall, precision))
                        temp_auroc.append(roc_auc_score(validation_outcome_true_vals[targ][bool_vec], val_outcome_preds_over_iters.iloc[:,i][bool_vec]))

                    if len(temp_auroc) == 1 or len(temp_auprc) == 1:
                        val_AUROC_20_sd = 0
                        val_AUPRC_20_sd = 0
                    else:
                        val_AUROC_20_sd = np.std(temp_auroc, ddof=1)
                        val_AUPRC_20_sd = np.std(temp_auprc, ddof=1)
                    val_AUROC_20_mean = np.mean(temp_auroc)
                    val_AUPRC_20_mean = np.mean(temp_auprc)

                    # Save results over iters
                    preds_iters_top_subgroups = (val_outcome_preds_over_iters[bool_vec]
                        .reset_index()
                        .melt(id_vars='row_id', value_name='preds')
                        .set_index('row_id'))
                    top_subgroups_iters_df = (pd.merge(
                        preds_iters_top_subgroups,
                        true_vals_top_subgroups.rename('true_vals'),
                        left_index=True, right_index=True))
                    top_subgroups_iters_df['outcome'] = targ
                    top_subgroups_iters_df['evaluation_metric'] = metric
                    top_subgroups_iters_df['dataset'] = 'holdout_validation'
                    top_k_subgroup_preds_iters.append(top_subgroups_iters_df)


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
        workbook = xlsxwriter.Workbook(output_dir + metric + "_bottleneck_results.xlsx")
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
                    if(val != val): #nan check
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
                    if(val != val):
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
            for row_num in range(len(precision_20)):
                temp = worksheet_pr.write(row_num+1,0,precision_20[row_num])
                temp = worksheet_pr.write(row_num+1,1,recall_20[row_num])
            for row_num in range(len(precision_val_20)):
                temp = worksheet_pr_val.write(row_num+1,0,precision_val_20[row_num])
                temp = worksheet_pr_val.write(row_num+1,1,recall_val_20[row_num])

            # adding ROC tpr and fpr axes
            worksheet_roc.write(0,0, "TPR KFold")
            worksheet_roc.write(0,1, "FPR KFold")
            worksheet_roc_val.write(0,0, "TPR Val")
            worksheet_roc_val.write(0,1, "FPR Val")
            for row_num in range(len(tpr_20)):
                temp = worksheet_roc.write(row_num+1,0,tpr_20[row_num])
                temp = worksheet_roc.write(row_num+1,1,fpr_20[row_num])
            for row_num in range(len(tpr_val_20)):
                temp = worksheet_roc_val.write(row_num+1,0,tpr_val_20[row_num])
                temp = worksheet_roc_val.write(row_num+1,1,fpr_val_20[row_num])

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
