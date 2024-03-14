# general imports
import argparse
import logging
import pandas as pd
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
    return parser.parse_args()


def main(args):
    results_dir = args.input_directory
    valid_metab_file = args.validation_metabolites
    valid_preds_file = args.validation_predictions
    valid_true_vals_file = args.validation_true_vals
    output_dir = args.output_directory
    val_index_col = args.validation_id

    # Set up logger
    # TODO: Set up loglevel
    logger = logging.getLogger('SubgroupDiscovery')

    # read in previous outputs from the bottleneck layer
    preds = pd.read_csv(results_dir + "bottleneck.csv")
    preds = preds.rename(columns={"Unnamed: 0":"row_id"})
    true_vals = pd.read_csv(results_dir + "true_vals.csv").set_index('row_id')

    # Read in Mednax predictions and true values
    # TODO: Read in Mednax metabolites
    val_preds = pd.read_csv(valid_preds_file).set_index(val_index_col)
    valid_true_vals = pd.read_csv(valid_true_vals_file).set_index(val_index_col)

    #read in raw data to get actual response for validation data, not currently included in prediction .csv's
    cal_biobank_data = pd.read_csv("./data/processed/neonatal_conditions.csv", low_memory=False)

    #maintain predictions across individual model runs to calculate Mean + SD
    preds_over_iters = preds.copy().pivot_table(index="row_id",columns="iter",values="bottleneck_unit_0")
    preds = preds.groupby(["row_id"])["bottleneck_unit_0"].mean() #smart way (it is, double checked against stupid for loop method)

    if 'iter' in true_vals.columns:
        true_vals = true_vals[true_vals.iter == 0].drop(
            columns=['fold', 'iter'])
    true_vals = true_vals.loc[preds.index, :]

    #collapse all outcomes to patients x outcomes dataframe
    external_true_vals = cal_biobank_data[["nec_any","rop_any","bpd_any", "ivh_any"]]
    metabolite_labels = pd.read_csv("./config/metabolite_labels.csv")

    # Read in the California metabolite labels
    with open('./config/expected_metabolite_order.txt') as f:
        cal_metabolites = [l.strip() for l in f.readlines()]

    # Read in metadata
    metadata = pd.read_csv("./data/processed/metadata.csv", low_memory=False)

    # Improve the ID setting, should be able to use the row_id instead of gdspid
    #subset by observations
    subset_data = metadata[metadata["row_id"].isin(pd.unique(preds.index))]
    subset_data = subset_data.set_index("row_id")
    subset_data = subset_data.drop("gdspid", axis=1)

    # validation_true_vals = external_true_vals.loc[subset_val_data.index]
    validation_true_vals = valid_true_vals.loc[val_preds.index]

    #check that all indices are the same
    assert (preds.index == subset_data.index).all()
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
            in_analysis_set = (true_vals[np.setdiff1d(["nec_any","rop_any","bpd_any","ivh_any"], [targ])].sum(axis=1) == 0) | (true_vals[targ] == 1)
            in_analysis_set_val = (validation_true_vals[np.setdiff1d(["nec_any","rop_any","bpd_any","ivh_any"], [targ])].sum(axis=1) == 0) | (validation_true_vals[targ] == 1)

            #k-fold
            outcome_preds = preds.loc[in_analysis_set]
            outcome_true_vals = true_vals.loc[in_analysis_set,:]
            outcome_subset_data = subset_data.loc[in_analysis_set,:]

            #validation
            val_outcome_preds = val_preds.loc[in_analysis_set_val]
            validation_outcome_true_vals = validation_true_vals.loc[in_analysis_set_val,:]

            #iter predictions to calculate SD
            many_outcome_preds = preds_over_iters.loc[in_analysis_set]
            many_val_outcome_preds = val_preds.loc[in_analysis_set_val]

            #double check that all indices are the same
            assert (preds.index == subset_data.index).all()
            assert (preds.index == true_vals.index).all()
            assert (val_preds.index == validation_true_vals.index).all()

            temp_data = outcome_subset_data[outcome_subset_data.columns.values[outcome_subset_data.isna().sum() == 0]].copy()

            #constructing list of demographic and metabolomic features to in_analysis_set
            # TODO: Change the logic for defining the columns to be clearer and more consistent with selected columns
            in_analysis_set_features = temp_data.columns.to_series().apply(lambda z: True if ("rc" in z) or (z in [targ+col_annotation]) else False)
            temp_data = temp_data[temp_data.columns[in_analysis_set_features]]

            #compile list of features which need to be transformed into quantiles
            transform = temp_data.apply(lambda z: z.name if (temp_data[z.name].dtype == "float64") or ("rc" in z.name) else None).unique()[1:]

            searchspace_data = {}

            #transform all data into various quantiles [2,3,5] as per martins experiment
            for c in temp_data.columns:
                if c in transform:
                    for q in [2,3,5]:
                        column = f"{c}_q-{q}"
                        searchspace_data[column] = pd.qcut(temp_data[c], q, duplicates="drop").cat.codes
                else:
                    searchspace_data[c] = temp_data[c]

            searchspace_data = pd.DataFrame(searchspace_data)

            # TODO: Should change the logic for finding out which columsn are metabolites
            # (don't have to do the string matching with '_rc')
            # create indicator vector for metabolite columns by matching rc string, used for metabolite only analyses if necessary
            # TODO: Remove the old metabolite data which is currently commented out
            # is_metabolite = searchspace_data.columns.to_series().apply(lambda z: True if "rc" in z else False)
            is_metabolite = searchspace_data.columns.isin(cal_metabolites)

            # TODO: Change the logic so that the metadata is not expected to be used in the subgroup discovery procedure
            temp_val_data = subset_val_data_outcome[subset_val_data_outcome.columns.values[subset_val_data_outcome.isna().sum() == 0]].copy()

            #constructing list of demographic and metabolomic features to in_analysis_set
            # FIXME: TODO: Fix the metabolite data selection logic
            # in_analysis_set_val_features = temp_val_data.columns.isin(cal_metabolites)
            in_analysis_set_val_features = temp_val_data.columns.to_series().apply(lambda z: True if ("rc" in z) or (z in [targ+col_annotation]) else False)
            temp_val_data = temp_val_data[temp_val_data.columns[in_analysis_set_val_features]]

            #compile list of features which need to be transformed into quantiles
            transform = temp_val_data.apply(lambda z: z.name if (temp_val_data[z.name].dtype == "float64") or ("rc" in z.name) else None).unique()[1:]

            searchspace_val_data = {}
            #transform all data into various quantiles [2,3,5] as per martins experiment
            for c in temp_val_data.columns:
                if c in transform:
                    for q in [2,3,5]:
                        column = f"{c}_q-{q}"
                        searchspace_val_data[column] = pd.qcut(temp_val_data[c], q, duplicates="drop").cat.codes
                else:
                    searchspace_val_data[c] = temp_val_data[c]


            searchspace_val_data = pd.DataFrame(searchspace_val_data)

            # TODO: Replace searching logic from '_rc' string. 
            # create indicator vector for metabolite columns by matching rc string, used for metabolite only analyses if necessary
            is_val_metabolite = searchspace_val_data.columns.to_series().apply(lambda z: True if "rc" in z else False)
 
            # TODO: When should the switch happen for the label 
            # TODO: NOTE: This was because previously this is used as a way to measure True Positives
            # TODO: The label switch DOES need to happen for calculating AUROC and AUPRC
            # NOTE: Since we are interested in identifying HEALTHY individuals
            # The target for prediction will be switched to healthy obs
            true_vals = 1 - true_vals # this along with a change in the 'in_analysis_set' vector is the only change
            validation_true_vals = 1 - validation_true_vals

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

            results= ps.BeamSearch(beam_width=subgroup_sizes[outcome]).execute(task)
            # TODO: Extract the subgroup discovery logic and apply to the Mednax dataset
            # TODO: Remove commented examples from the final script
            # EXAMPLES
            # pickle after removing data, if you had to.
            # results.task.data = None
            # pickle.dump(...)
            # results to dataframe (hard to add custom statistics)
            results.to_dataframe()
            results.results # all the subgroups
            # get first subgroup
            sg_quality, sg_description, qf = results.results[0]
            # get subgroup mask / selection array
            sg_msk, sg_size = ps.get_cover_array_and_size(sg_description, data=searchspace_data)

            # NOTE: This is where the original logic resumes.
            subgroup_desc = results.to_dataframe()["subgroup"]

            data = []
            total_elem = ""
            bool_vec = np.full((len(searchspace_data.index)), False)
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
                #collect same general data for top 1:index(elem) subgroups

                # NOTE: The next logic exists because it's difficult to add custom stats to the dataframe original output
                precision, recall, thresholds = precision_recall_curve(outcome_true_vals[targ][bool_vec], outcome_preds[bool_vec])
                AUPRC = auc(recall, precision)
                precision, recall, thresholds = precision_recall_curve(outcome_true_vals[targ][bool_vec_inner], outcome_preds[bool_vec_inner])
                subgroup_AUPRC = auc(recall, precision)
                #append all data to array
                #compile metrics of performancpe
                if(len(outcome_true_vals[targ][bool_vec]) == outcome_true_vals[targ][bool_vec].sum()):
                    AUROC = np.nan
                    AUROC_sd = np.nan
                    AUROC_mean = np.nan

                    # Loops like this collect performance of individual model iterations
                    temp_auprc = []
                    for i in range(len(preds_over_iters.columns)):
                        precision, recall, thresholds = precision_recall_curve(outcome_true_vals[targ][bool_vec], many_outcome_preds.iloc[:,i][bool_vec])
                        temp_auprc.append(auc(recall, precision))

                    AUPRC_sd = np.std(temp_auprc, ddof=1) #used to make std consistent with R
                    AUPRC_mean = np.mean(temp_auprc)

                else:
                    AUROC = roc_auc_score(outcome_true_vals[targ][bool_vec], outcome_preds[bool_vec])

                    temp_auroc = []
                    temp_auprc = []
                    for i in range(len(preds_over_iters.columns)):
                        precision, recall, thresholds = precision_recall_curve(outcome_true_vals[targ][bool_vec], many_outcome_preds.iloc[:,i][bool_vec])
                        temp_auprc.append(auc(recall, precision))


                    AUROC_sd = np.std(temp_auroc, ddof=1)
                    AUROC_mean = np.mean(temp_auroc)
                    AUPRC_sd = np.std(temp_auprc, ddof=1)
                    AUPRC_mean = np.mean(temp_auprc)


                if(len(outcome_true_vals[targ][bool_vec_inner]) == outcome_true_vals[targ][bool_vec_inner].sum()):
                    subgroup_AUROC = np.nan
                    subgroup_AUROC_sd = np.nan
                    subgroup_AUROC_mean = np.nan

                    temp_auprc = []
                    for i in range(len(preds_over_iters.columns)):
                        precision, recall, thresholds = precision_recall_curve(outcome_true_vals[targ][bool_vec_inner], many_outcome_preds.iloc[:,i][bool_vec_inner])
                        temp_auprc.append(auc(recall, precision))

                    subgroup_AUPRC_sd = np.std(temp_auprc, ddof=1)
                    subgroup_AUPRC_mean = np.mean(temp_auprc)

                else:
                    subgroup_AUROC = roc_auc_score(outcome_true_vals[targ][bool_vec_inner], outcome_preds[bool_vec_inner])


                    temp_auroc = []
                    temp_auprc = []
                    for i in range(len(preds_over_iters.columns)):
                        precision, recall, thresholds = precision_recall_curve(outcome_true_vals[targ][bool_vec_inner], many_outcome_preds.iloc[:,i][bool_vec_inner])
                        temp_auprc.append(auc(recall, precision))
                        temp_auroc.append(roc_auc_score(outcome_true_vals[targ][bool_vec_inner], many_outcome_preds.iloc[:,i][bool_vec_inner]))

                    subgroup_AUROC_sd = np.std(temp_auroc, ddof=1)
                    subgroup_AUROC_mean = np.mean(temp_auroc)
                    subgroup_AUPRC_sd = np.std(temp_auprc, ddof=1)
                    subgroup_AUPRC_mean = np.mean(temp_auprc)

                total_elem = elem + "-OR-" + total_elem
                data.append([total_elem, elem, bool_vec.sum(), bool_vec.sum()/len(searchspace_data.iloc[:,0]), bool_vec_inner.sum(), count, AUPRC, subgroup_AUPRC, AUROC, subgroup_AUROC])

            subgroup_results_df = pd.DataFrame(data, columns=["total group", "subgroup", "size", "% data", "subgroup size", "num_groups", "AUPRC","subgroup AUPRC", "AUROC", "subgroup AUROC"])

            #compile metrics of performancpe
            kfold_AUROC = roc_auc_score(outcome_true_vals[targ], outcome_preds)
            precision, recall, thresholds = precision_recall_curve(outcome_true_vals[targ], outcome_preds)
            kfold_AUPRC = auc(recall, precision)

            temp_auroc = []
            temp_auprc = []
            for i in range(len(preds_over_iters.columns)):
                precision, recall, thresholds = precision_recall_curve(outcome_true_vals[targ], many_outcome_preds.iloc[:,i])
                temp_auprc.append(auc(recall, precision))
                temp_auroc.append(roc_auc_score(outcome_true_vals[targ], many_outcome_preds.iloc[:,i]))

            kfold_AUROC_sd = np.std(temp_auroc, ddof=1)
            kfold_AUROC_mean = np.mean(temp_auroc)
            kfold_AUPRC_sd = np.std(temp_auprc, ddof=1)
            kfold_AUPRC_mean = np.mean(temp_auprc)

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
                precision, recall, thresholds = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec_inner], val_outcome_preds[bool_vec_inner])
                subgroup_AUPRC = auc(recall, precision)


                if(len(validation_outcome_true_vals[targ][bool_vec]) == validation_outcome_true_vals[targ][bool_vec].sum()):
                    AUROC = np.nan
                    AUROC_sd = np.nan
                    AUROC_mean = np.nan

                    temp_auprc = []
                    for i in range(len(preds_over_iters.columns)):
                        precision, recall, thresholds = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec], many_val_outcome_preds.iloc[:,i][bool_vec])
                        temp_auprc.append(auc(recall, precision))

                    AUPRC_sd = np.std(temp_auprc, ddof=1)
                    AUPRC_mean = np.mean(temp_auprc)

                else:
                    AUROC = roc_auc_score(validation_outcome_true_vals[targ][bool_vec], val_outcome_preds[bool_vec])

                    temp_auroc = []
                    temp_auprc = []
                    for i in range(len(preds_over_iters.columns)):
                        precision, recall, thresholds = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec], many_val_outcome_preds.iloc[:,i][bool_vec])
                        temp_auprc.append(auc(recall, precision))
                        temp_auroc.append(roc_auc_score(validation_outcome_true_vals[targ][bool_vec], many_val_outcome_preds.iloc[:,i][bool_vec]))

                    AUROC_sd = np.std(temp_auroc, ddof=1)
                    AUROC_mean = np.mean(temp_auroc)
                    AUPRC_sd = np.std(temp_auprc, ddof=1)
                    AUPRC_mean = np.mean(temp_auprc)



                if(len(validation_outcome_true_vals[targ][bool_vec_inner]) == validation_outcome_true_vals[targ][bool_vec_inner].sum()):
                    subgroup_AUROC = np.nan
                    subgroup_AUROC_sd = np.nan
                    subgroup_AUROC_mean = np.nan

                    temp_auprc = []
                    for i in range(len(preds_over_iters.columns)):
                        precision, recall, thresholds = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec_inner], many_val_outcome_preds.iloc[:,i][bool_vec_inner])
                        temp_auprc.append(auc(recall, precision))

                    subgroup_AUPRC_sd = np.std(temp_auprc, ddof=1)
                    subgroup_AUPRC_mean = np.mean(temp_auprc)

                else:
                    subgroup_AUROC = roc_auc_score(validation_outcome_true_vals[targ][bool_vec_inner], val_outcome_preds[bool_vec_inner])

                    #calculate sd metrics
                    temp_auroc = []
                    temp_auprc = []
                    for i in range(len(preds_over_iters.columns)):
                        precision, recall, thresholds = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec_inner], many_val_outcome_preds.iloc[:,i][bool_vec_inner])
                        temp_auprc.append(auc(recall, precision))
                        temp_auroc.append(roc_auc_score(validation_outcome_true_vals[targ][bool_vec_inner], many_val_outcome_preds.iloc[:,i][bool_vec_inner]))

                    subgroup_AUROC_sd = np.std(temp_auroc, ddof=1)
                    subgroup_AUROC_mean = np.mean(temp_auroc)
                    subgroup_AUPRC_sd = np.std(temp_auprc, ddof=1)
                    subgroup_AUPRC_mean = np.mean(temp_auprc)


                total_elem = elem + "-OR-" + total_elem
                data.append([total_elem, elem, bool_vec.sum(), bool_vec.sum()/len(searchspace_val_data.iloc[:,0]), bool_vec_inner.sum(), count, AUPRC, subgroup_AUPRC, AUROC, subgroup_AUROC])

            subgroup_val_results_df = pd.DataFrame(data, columns=["total group","subgroup","size", "% data", "subgroup size", "num_groups", "AUPRC", "subgroup AUPRC", "AUROC", "subgroup AUROC"])

            val_AUROC = roc_auc_score(validation_outcome_true_vals[targ], val_outcome_preds)
            precision, recall, thresholds = precision_recall_curve(validation_outcome_true_vals[targ], val_outcome_preds)
            val_AUPRC = auc(recall, precision)

            temp_auroc = []
            temp_auprc = []
            for i in range(len(preds_over_iters.columns)):
                precision1, recall1, thresholds = precision_recall_curve(validation_outcome_true_vals[targ], many_val_outcome_preds.iloc[:,i])
                temp_auprc.append(auc(recall1, precision1))
                temp_auroc.append(roc_auc_score(validation_outcome_true_vals[targ], many_val_outcome_preds.iloc[:,i]))

            val_AUROC_sd = np.std(temp_auroc, ddof=1)
            val_AUROC_mean = np.mean(temp_auroc)
            val_AUPRC_sd = np.std(temp_auprc, ddof=1)
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
                    for i in range(len(preds_over_iters.columns)):
                        precision, recall, thresholds = precision_recall_curve(outcome_true_vals[targ][bool_vec], many_outcome_preds.iloc[:,i][bool_vec])
                        temp_auprc.append(auc(recall, precision))
                        temp_auroc.append(roc_auc_score(outcome_true_vals[targ][bool_vec], many_outcome_preds.iloc[:,i][bool_vec]))

                    kfold_AUROC_20_sd = np.std(temp_auroc, ddof=1)
                    kfold_AUROC_20_mean = np.mean(temp_auroc)
                    kfold_AUPRC_20_sd = np.std(temp_auprc, ddof=1)
                    kfold_AUPRC_20_mean = np.mean(temp_auprc)

                    # Save predictions in top subgroups over iters
                    preds_iters_top_subgroups = (many_outcome_preds[bool_vec]
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
                    for i in range(len(preds_over_iters.columns)):
                        precision, recall, thresholds = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec], many_val_outcome_preds.iloc[:,i][bool_vec])
                        temp_auprc.append(auc(recall, precision))
                        temp_auroc.append(roc_auc_score(validation_outcome_true_vals[targ][bool_vec], many_val_outcome_preds.iloc[:,i][bool_vec]))

                    val_AUROC_20_sd = np.std(temp_auroc, ddof=1)
                    val_AUROC_20_mean = np.mean(temp_auroc)
                    val_AUPRC_20_sd = np.std(temp_auprc, ddof=1)
                    val_AUPRC_20_mean = np.mean(temp_auprc)

                    # Save results over iters
                    preds_iters_top_subgroups = (many_val_outcome_preds[bool_vec]
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
