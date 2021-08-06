##########################################################################
## This script generates all subgroup results for prediction prediction ##
##########################################################################
# importing some utilities functions
import argparse
import xlsxwriter
import pickle

# general imports
import pandas as pd
import numpy as np

# sklearn imports for evaluating results
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

#NOTE: altered pysubgroup package must be installed prior to analysis
#to install use the following command "pip install git+https://github.com/Teculos/pysubgroup.git@predictionQF"
import pysubgroup as ps

#setting printing parameters so subgroup descriptions actually display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 250)
pd.set_option('display.max_colwidth', 150)

# Argument parsing
def get_args():
    parser = argparse.ArgumentParser(
        description='Perform subgroup discovery on model predictions',
        conflict_handler='resolve')
    parser.add_argument(
        '-i', '--input_directory',
        help='Input experiment directory, should contain preds.csv.gz, valid_preds.csv, true_vals.csv')
    parser.add_argument(
        '-o', '--output_directory',
        help='output directory to save results files')
    return parser.parse_args()

def main(args):
    ###################################################################################
    ## Read in of preds, true values, and data used for subgroup disc (metadata.csv) ##
    ###################################################################################

    outcome_order = ["bpd", "rop", "ivh", "nec"]
    results_dir = args.input_directory
    output_dir = args.output_directory

    # read in previous predicted values
    preds = pd.read_csv(results_dir + "preds.csv.gz")
    valid_preds = pd.read_csv(results_dir + "valid_preds.csv")
    true_vals = pd.read_csv(results_dir + "true_vals.csv")

    #read in raw data to get actual response for validation data, not currently included in prediction .csv's
    external_true_vals = pd.read_csv("./data/processed/neonatal_conditions.csv", low_memory=False)

    #remane pred columns to be consistent
    valid_preds = valid_preds.rename(columns={"Unnamed: 0":"row_id"})
    preds = preds.rename(columns={"Unnamed: 0":"row_id"})

    #maintain predictions across individual model runs to calculate Mean + SD
    many_preds = {x:preds.copy().pivot_table(index="row_id",columns="iter", values=x+"_any") for x in outcome_order}
    many_valid_preds = {x:valid_preds.copy().pivot_table(index="row_id",columns="iter", values=x+"_any") for x in outcome_order}

    #average over all iteration runs, previously was only taking one
    preds = preds.groupby(["row_id"])[["nec_any","rop_any","bpd_any", "ivh_any"]].mean() #smart way (it is, double checked against stupid for loop method)
    valid_preds = valid_preds.groupby(["row_id"])[["nec_any","rop_any","bpd_any", "ivh_any"]].mean() #smart way (it is, double checked against stupid for loop method)
    true_vals = true_vals.groupby(["row_id"])[["nec_any","rop_any","bpd_any", "ivh_any"]].mean() #smart way (it is, double checked against stupid for loop method)

    #collapse all outcomes to patients x outcomes dataframe
    external_true_vals = external_true_vals[["nec_any","rop_any","bpd_any", "ivh_any"]]

    metabolite_labels = pd.read_csv("./config/metabolite_labels.csv")
    data = pd.read_csv("./data/processed/metadata.csv", low_memory=False)

    #######################################################
    ## reduce *_data and *_values to only predicted data ##
    #######################################################

    #subset by observations
    subset_data = data[data["row_id"].isin(pd.unique(preds.index))]
    subset_data = subset_data.set_index("row_id")
    subset_data = subset_data.drop("gdspid", axis=1)

    #subset by observations
    subset_val_data = data[data["row_id"].isin(pd.unique(valid_preds.index))]
    subset_val_data = subset_val_data.set_index("row_id")
    subset_val_data = subset_val_data.drop("gdspid", axis=1)

    #subsetting  holdout predictions
    validation_true_vals = external_true_vals.loc[subset_val_data.index]

    ###############################
    ## Align all data structures ##
    ###############################

    #check that all indices are the same
    assert (preds.index == subset_data.index).all()
    assert (preds.index == true_vals.index).all()
    assert (preds.index == many_preds[outcome_order[0]].index).all()

    assert (valid_preds.index == subset_val_data.index).all()
    assert (valid_preds.index == validation_true_vals.index).all()
    assert (valid_preds.index == many_valid_preds[outcome_order[0]].index).all()

    ########################################################################
    ## manipulate predicted probs into class labels via different cutoffs ##
    ########################################################################
    #order is bpd -> rop -> ivh -> nec
    evaluation_order = ["AUROC", "AVG Precision"]

    subgroup_alphas_avg_prec = {"bpd":0.075, "rop":0.0648, "ivh":0.076, "nec":0.14}
    subgroup_sizes_avg_prec = {"bpd":100, "rop":150, "ivh":150, "nec":100}

    subgroup_alphas_auroc = {"bpd":0.065, "rop":0.07, "ivh":0.065, "nec":0.069}
    subgroup_sizes_auroc = {"bpd":100, "rop":200, "ivh":100, "nec":100}

    subgroup_alphas_list = {"AVG Precision": subgroup_alphas_avg_prec, "AUROC": subgroup_alphas_auroc}
    subgroup_sizes_list = {"AVG Precision": subgroup_sizes_avg_prec, "AUROC": subgroup_sizes_auroc}
    evaluation_lists = {"AVG Precision": average_precision_score, "AUROC": roc_auc_score}

    # Create a list of dataframes for the top K predictions from each subgroup discovery setting
    top_k_subgroup_predictions = []
    # This second list stores predictions from individual iterations
    top_k_subgroup_preds_iters = []

    for metric in evaluation_order:
        print("starting analysis using - " + metric)
        #
        subgroup_alphas = subgroup_alphas_list[metric]
        subgroup_sizes = subgroup_sizes_list[metric]
        evaluation_metric = evaluation_lists[metric]
        #
        all_results = {}
        iter_results = {}

        #start of loop to collect all data
        for outcome in outcome_order:
            print("starting analysis of - " + outcome)
            #this will create the target matrix, possible targets include true classification, true positivies, ect (based on a CUTOFF value)
            targ = outcome+"_any"
            pred_type = "_tp"
            #
            #limiting to TRUE healthy controls (removing controls with positive co-outcomes)
            keep = (true_vals[np.setdiff1d(["nec_any","rop_any","bpd_any","ivh_any"], [targ])].sum(axis=1) == 0) | (true_vals[targ] == 1)
            keep_val = (validation_true_vals[np.setdiff1d(["nec_any","rop_any","bpd_any","ivh_any"], [targ])].sum(axis=1) == 0) | (validation_true_vals[targ] == 1)
            #
            #k-fold
            outcome_preds = preds.loc[keep]
            outcome_true_vals = true_vals.loc[keep,:]
            subset_data_outcome = subset_data.loc[keep,:]
            #
            #validation
            val_outcome_preds = valid_preds.loc[keep_val]
            validation_outcome_true_vals = validation_true_vals.loc[keep_val,:]
            subset_val_data_outcome = subset_val_data.loc[keep_val,:]
            #
            #iter predictions to calculate SD
            many_outcome_preds = many_preds[outcome].loc[keep]
            many_val_outcome_preds = many_valid_preds[outcome].loc[keep_val]
            #
            # Double check that all indices are the same?
            assert (preds.index == subset_data.index).all()
            assert (preds.index == true_vals.index).all()
            assert (valid_preds.index == subset_val_data.index).all()
            assert (valid_preds.index == validation_true_vals.index).all()
            #
            ###########################################################################
            ## massage metabolites and demographic data into quantiles for discovery ##
            ###########################################################################
            temp_data = subset_data_outcome[subset_data_outcome.columns.values[subset_data_outcome.isna().sum() == 0]].copy()
            #
            #constructing list of demographic and metabolomic features to keep
            keep_features = temp_data.columns.to_series().apply(lambda z: True if ("rc" in z) or (z in [targ+pred_type]) else False)
            temp_data = temp_data[temp_data.columns[keep_features]]
            #
            #compile list of features which need to be transformed into quantiles
            transform = temp_data.apply(lambda z: z.name if (temp_data[z.name].dtype == "float64") or ("rc" in z.name) else None).unique()[1:]
            #
            searchspace_data = {}
            #
            #transform all data into various quantiles [2,3,5]
            for c in temp_data.columns:
                if c in transform:
                    for q in [2,3,5]:
                        column = f"{c}_q-{q}"
                        searchspace_data[column] = pd.qcut(temp_data[c], q, duplicates="drop").cat.codes
                else:
                    searchspace_data[c] = temp_data[c]
            #
            #
            searchspace_data = pd.DataFrame(searchspace_data)
            # create indicator vector for metabolite columns by matching rc string, used for metabolite only analyses if necessary
            is_metabolite = searchspace_data.columns.to_series().apply(lambda z: True if "rc" in z else False)
            #
            #########################################
            ## Massaging validation data similarly ##
            #########################################
            #
            temp_val_data = subset_val_data_outcome[subset_val_data_outcome.columns.values[subset_val_data_outcome.isna().sum() == 0]].copy()
            #
            #constructing list of demographic and metabolomic features to keep
            keep_val_features = temp_val_data.columns.to_series().apply(lambda z: True if ("rc" in z) or (z in [targ+pred_type]) else False)
            temp_val_data = temp_val_data[temp_val_data.columns[keep_val_features]]
            #
            #compile list of features which need to be transformed into quantiles
            transform = temp_val_data.apply(lambda z: z.name if (temp_val_data[z.name].dtype == "float64") or ("rc" in z.name) else None).unique()[1:]
            #
            searchspace_val_data = {}
            #transform all data into various quantiles [2,3,5] as per martins experiment
            for c in temp_val_data.columns:
                if c in transform:
                    for q in [2,3,5]:
                        column = f"{c}_q-{q}"
                        searchspace_val_data[column] = pd.qcut(temp_val_data[c], q, duplicates="drop").cat.codes
                else:
                    searchspace_val_data[c] = temp_val_data[c]
            #
            #
            searchspace_val_data = pd.DataFrame(searchspace_val_data)
            #
            # create indicator vector for metabolite columns by matching rc string, used for metabolite only analyses if necessary
            is_val_metabolite = searchspace_val_data.columns.to_series().apply(lambda z: True if "rc" in z else False)
            #
            ###################################################################
            ## initial subgroup analysis using sklearn evaluations as target ##
            ###################################################################
            #
            target = ps.PredictionTarget(outcome_true_vals[targ].to_numpy(), outcome_preds[targ].to_numpy(), evaluation_metric)
            searchspace = ps.create_selectors(searchspace_data[searchspace_data.columns[is_metabolite]])
            task = ps.SubgroupDiscoveryTask(
                searchspace_data,
                target,
                searchspace,
                result_set_size=subgroup_sizes[outcome],
                depth=4,
                qf=ps.PredictionQFNumeric(a=subgroup_alphas[outcome])
            )
            #
            results = ps.BeamSearch(beam_width=subgroup_sizes[outcome]).execute(task)
            #
            # iterate over and combine subgroups to extract information regarding subgroup predictive performance
            subgroup_desc = results.to_dataframe()["subgroup"]

            # Parse subgroup descriptions to iterate through individual subgroups
            # To calculate WITHIN-subgroup AUROC and AUPR
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
                precision, recall, thresholds = precision_recall_curve(outcome_true_vals[targ][bool_vec], outcome_preds[targ][bool_vec])
                AUPRC = auc(recall, precision)
                #
                precision, recall, thresholds = precision_recall_curve(outcome_true_vals[targ][bool_vec_inner], outcome_preds[targ][bool_vec_inner])
                subgroup_AUPRC = auc(recall, precision)
                #
                #
                if(len(outcome_true_vals[targ][bool_vec]) == outcome_true_vals[targ][bool_vec].sum()):
                    AUROC = np.nan
                    AUROC_sd = np.nan
                    AUROC_mean = np.nan
                    #
                    # TODO: Question: Do loops like this collect performance of individual model iterations?
                    temp_auprc = []
                    for i in range(len(many_outcome_preds.columns)):
                        precision, recall, thresholds = precision_recall_curve(outcome_true_vals[targ][bool_vec], many_outcome_preds.iloc[:,i][bool_vec])
                        temp_auprc.append(auc(recall, precision))
                    #
                    AUPRC_sd = np.std(temp_auprc, ddof=1) #used to make std consistent with R
                    AUPRC_mean = np.mean(temp_auprc)
                    #
                else:
                    AUROC = roc_auc_score(outcome_true_vals[targ][bool_vec], outcome_preds[targ][bool_vec])
                    #
                    temp_auroc = []
                    temp_auprc = []
                    for i in range(len(many_outcome_preds.columns)):
                        precision, recall, thresholds = precision_recall_curve(outcome_true_vals[targ][bool_vec], many_outcome_preds.iloc[:,i][bool_vec])
                        temp_auprc.append(auc(recall, precision))
                        temp_auroc.append(roc_auc_score(outcome_true_vals[targ][bool_vec], many_outcome_preds.iloc[:,i][bool_vec])) #
                    #
                    AUROC_sd = np.std(temp_auroc, ddof=1)
                    AUROC_mean = np.mean(temp_auroc)
                    AUPRC_sd = np.std(temp_auprc, ddof=1)
                    AUPRC_mean = np.mean(temp_auprc)
                #
                #
                #
                if(len(outcome_true_vals[targ][bool_vec_inner]) == outcome_true_vals[targ][bool_vec_inner].sum()):
                    subgroup_AUROC = np.nan
                    subgroup_AUROC_sd = np.nan
                    subgroup_AUROC_mean = np.nan
                    #
                    temp_auprc = []
                    for i in range(len(many_outcome_preds.columns)):
                        precision, recall, thresholds = precision_recall_curve(outcome_true_vals[targ][bool_vec_inner], many_outcome_preds.iloc[:,i][bool_vec_inner])
                        temp_auprc.append(auc(recall, precision))
                    #
                    subgroup_AUPRC_sd = np.std(temp_auprc, ddof=1)
                    subgroup_AUPRC_mean = np.mean(temp_auprc)
                    #
                else:
                    subgroup_AUROC = roc_auc_score(outcome_true_vals[targ][bool_vec_inner], outcome_preds[targ][bool_vec_inner])
                    #
                    ##
                    temp_auroc = []
                    temp_auprc = []
                    for i in range(len(many_outcome_preds.columns)):
                        precision, recall, thresholds = precision_recall_curve(outcome_true_vals[targ][bool_vec_inner], many_outcome_preds.iloc[:,i][bool_vec_inner])
                        temp_auprc.append(auc(recall, precision))
                        temp_auroc.append(roc_auc_score(outcome_true_vals[targ][bool_vec_inner], many_outcome_preds.iloc[:,i][bool_vec_inner]))
                    #
                    subgroup_AUROC_sd = np.std(temp_auroc, ddof=1)
                    subgroup_AUROC_mean = np.mean(temp_auroc)
                    subgroup_AUPRC_sd = np.std(temp_auprc, ddof=1)
                    subgroup_AUPRC_mean = np.mean(temp_auprc)
                #
                #collect results for this outcome to be found in 'subgroup_results_df'
                total_elem = elem + "-OR-" + total_elem
                data.append([total_elem, elem, bool_vec.sum(), bool_vec.sum()/len(searchspace_data.iloc[:,0]), bool_vec_inner.sum(), count, AUPRC, subgroup_AUPRC, AUROC, subgroup_AUROC])
            #
            #
            subgroup_results_df = pd.DataFrame(data, columns=["total group", "subgroup", "size", "% data", "subgroup size", "num_groups", "AUPRC","subgroup AUPRC", "AUROC", "subgroup AUROC"])
            #
            #
            #compile metrics of performancpe
            kfold_AUROC = roc_auc_score(outcome_true_vals[targ], outcome_preds[targ])
            precision, recall, thresholds = precision_recall_curve(outcome_true_vals[targ], outcome_preds[targ])
            kfold_AUPRC = auc(recall, precision)
            #
            temp_auroc = []
            temp_auprc = []
            for i in range(len(many_outcome_preds.columns)):
                precision, recall, thresholds = precision_recall_curve(outcome_true_vals[targ], many_outcome_preds.iloc[:,i])
                temp_auprc.append(auc(recall, precision))
                temp_auroc.append(roc_auc_score(outcome_true_vals[targ], many_outcome_preds.iloc[:,i]))
            #
            kfold_AUROC_sd = np.std(temp_auroc, ddof=1)
            kfold_AUROC_mean = np.mean(temp_auroc)
            kfold_AUPRC_sd = np.std(temp_auprc, ddof=1)
            kfold_AUPRC_mean = np.mean(temp_auprc)
            #
            #
            ################################################################################
            ## Collect AUROC, AUPRC, and other metrics on held out validation predictions ##
            ################################################################################
            #
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
                #
                precision, recall, thresholds = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec], val_outcome_preds[targ][bool_vec])
                AUPRC = auc(recall, precision)
                #
                if(validation_outcome_true_vals[targ][bool_vec_inner].sum() !=0):
                    precision, recall, thresholds = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec_inner], val_outcome_preds[targ][bool_vec_inner])
                    subgroup_AUPRC = auc(recall, precision)
                else:
                    subgroup_AUPRC = np.nan
                    #
                #
                if(len(validation_outcome_true_vals[targ][bool_vec]) == validation_outcome_true_vals[targ][bool_vec].sum()):
                    AUROC = np.nan
                    AUROC_sd = np.nan
                    AUROC_mean = np.nan
                    #
                    temp_auprc = []
                    for i in range(len(many_outcome_preds.columns)):
                        precision, recall, thresholds = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec], many_val_outcome_preds.iloc[:,i][bool_vec])
                        temp_auprc.append(auc(recall, precision))
                    #
                    AUPRC_sd = np.std(temp_auprc, ddof=1)
                    AUPRC_mean = np.mean(temp_auprc)
                    #
                else:
                    AUROC = roc_auc_score(validation_outcome_true_vals[targ][bool_vec], val_outcome_preds[targ][bool_vec])
                    #
                    temp_auroc = []
                    temp_auprc = []
                    for i in range(len(many_outcome_preds.columns)):
                        precision, recall, thresholds = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec], many_val_outcome_preds.iloc[:,i][bool_vec])
                        temp_auprc.append(auc(recall, precision))
                        temp_auroc.append(roc_auc_score(validation_outcome_true_vals[targ][bool_vec], many_val_outcome_preds.iloc[:,i][bool_vec]))
                    #
                    AUROC_sd = np.std(temp_auroc, ddof=1)
                    AUROC_mean = np.mean(temp_auroc)
                    AUPRC_sd = np.std(temp_auprc, ddof=1)
                    AUPRC_mean = np.mean(temp_auprc)
                #
                #
                #
                if(len(validation_outcome_true_vals[targ][bool_vec_inner]) == len(validation_outcome_true_vals[targ][bool_vec_inner]) - validation_outcome_true_vals[targ][bool_vec_inner].sum() or len(validation_outcome_true_vals[targ][bool_vec_inner]) == validation_outcome_true_vals[targ][bool_vec_inner].sum()):
                    subgroup_AUROC = np.nan
                    subgroup_AUROC_sd = np.nan
                    subgroup_AUROC_mean = np.nan
                    #
                    if(len(validation_outcome_true_vals[targ][bool_vec_inner]) != len(validation_outcome_true_vals[targ][bool_vec_inner]) - validation_outcome_true_vals[targ][bool_vec_inner].sum()):
                        temp_auprc = []
                        for i in range(len(many_outcome_preds.columns)):
                            precision, recall, thresholds = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec_inner], many_val_outcome_preds.iloc[:,i][bool_vec_inner])
                            temp_auprc.append(auc(recall, precision))
                        #
                        subgroup_AUPRC_sd = np.std(temp_auprc, ddof=1)
                        subgroup_AUPRC_mean = np.mean(temp_auprc)
                    else:
                        subgroup_AUPRC_sd = np.nan
                        subgroup_AUPRC_mean = np.nan
                        #
                else:
                    subgroup_AUROC = roc_auc_score(validation_outcome_true_vals[targ][bool_vec_inner], val_outcome_preds[targ][bool_vec_inner])
                    #
                    #calculate sd metrics
                    temp_auroc = []
                    temp_auprc = []
                    for i in range(len(many_outcome_preds.columns)):
                        precision, recall, thresholds = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec_inner], many_val_outcome_preds.iloc[:,i][bool_vec_inner])
                        temp_auprc.append(auc(recall, precision))
                        temp_auroc.append(roc_auc_score(validation_outcome_true_vals[targ][bool_vec_inner], many_val_outcome_preds.iloc[:,i][bool_vec_inner]))
                    #
                    subgroup_AUROC_sd = np.std(temp_auroc, ddof=1)
                    subgroup_AUROC_mean = np.mean(temp_auroc)
                    subgroup_AUPRC_sd = np.std(temp_auprc, ddof=1)
                    subgroup_AUPRC_mean = np.mean(temp_auprc)
                #
                #
                total_elem = elem + "-OR-" + total_elem
                data.append([total_elem, elem, bool_vec.sum(), bool_vec.sum()/len(searchspace_val_data.iloc[:,0]), bool_vec_inner.sum(), count, AUPRC, subgroup_AUPRC, AUROC, subgroup_AUROC])
            #
            subgroup_val_results_df = pd.DataFrame(data, columns=["total group","subgroup","size", "% data", "subgroup size", "num_groups", "AUPRC", "subgroup AUPRC", "AUROC", "subgroup AUROC"])
            #
            val_AUROC = roc_auc_score(validation_outcome_true_vals[targ], val_outcome_preds[targ])
            precision, recall, thresholds = precision_recall_curve(validation_outcome_true_vals[targ], val_outcome_preds[targ])
            val_AUPRC = auc(recall, precision)
            #
            temp_auroc = []
            temp_auprc = []
            for i in range(len(many_outcome_preds.columns)):
                precision1, recall1, thresholds = precision_recall_curve(validation_outcome_true_vals[targ], many_val_outcome_preds.iloc[:,i])
                temp_auprc.append(auc(recall1, precision1))
                temp_auroc.append(roc_auc_score(validation_outcome_true_vals[targ], many_val_outcome_preds.iloc[:,i]))
            #
            val_AUROC_sd = np.std(temp_auroc, ddof=1)
            val_AUROC_mean = np.mean(temp_auroc)
            val_AUPRC_sd = np.std(temp_auprc, ddof=1)
            val_AUPRC_mean = np.mean(temp_auprc)
            #
            #create random vector
            np.random.seed(1234)
            rand_val_pred = np.random.uniform(0,1,len(validation_outcome_true_vals[targ]))
            rand_pred = np.random.uniform(0,1,len(outcome_true_vals[targ]))
            #
            rand_AUROC = roc_auc_score(outcome_true_vals[targ], rand_pred)
            rand_precision, rand_recall, rand_thresholds = precision_recall_curve(outcome_true_vals[targ], rand_pred)
            rand_AUPRC = auc(rand_recall, rand_precision)
            #
            rand_val_AUROC = roc_auc_score(validation_outcome_true_vals[targ], rand_val_pred)
            rand_precision, rand_recall, rand_thresholds = precision_recall_curve(validation_outcome_true_vals[targ], rand_val_pred)
            rand_val_AUPRC = auc(rand_recall, rand_precision)

            ################################################################################
            ## Calculate AUROC and AUPR at a specified cut-off for the percentile of data ##
            ################################################################################
            # Usually the top 20 percentile of data
            select =  (subgroup_results_df["% data"]* 100)
            select_index = select.index[select == min(select, key=lambda x:abs(x-20))][0]
            bool_vec = np.full((len(searchspace_data.index)), False)
            #
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
                    # Means: count variable has reached the percentile threshold for calculation
                    # Extract the true values and predictions from "outcome_true_vals[targ][bool_vec], outcome_preds[targ][bool_vec]"
                    preds_top_subgroups = outcome_preds[targ][bool_vec]
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

                    # Calculate ROC and AUPR on the top K% dataset
                    ROC_tuple_20 = roc_curve(true_vals_top_subgroups, preds_top_subgroups)
                    kfold_AUROC_20 = auc(ROC_tuple_20[0], ROC_tuple_20[1])
                    PR_tuple_20 = precision_recall_curve(true_vals_top_subgroups, preds_top_subgroups)
                    kfold_AUPRC_20 = auc(PR_tuple_20[1], PR_tuple_20[0])

                    # Compare with random prediction on the same identified subgroup
                    rand_AUROC_20 = roc_auc_score(true_vals_top_subgroups, random_preds_top_subgroups)
                    rand_precision, rand_recall, rand_thresholds = precision_recall_curve(true_vals_top_subgroups, random_preds_top_subgroups)
                    rand_AUPRC_20 = auc(rand_recall, rand_precision)

                    # I think this is the mean and sd across iterations of repeated K-Fold Cross Validation
                    temp_auroc = []
                    temp_auprc = []
                    for i in range(len(many_outcome_preds.columns)):
                        precision, recall, thresholds = precision_recall_curve(outcome_true_vals[targ][bool_vec], many_outcome_preds.iloc[:,i][bool_vec])
                        temp_auprc.append(auc(recall, precision))
                        temp_auroc.append(roc_auc_score(outcome_true_vals[targ][bool_vec], many_outcome_preds.iloc[:,i][bool_vec]))
                    #
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
                #
            #
            select =  (subgroup_val_results_df["% data"]* 100)
            select_index = select.index[select == min(select, key=lambda x:abs(x-20))][0]
            bool_vec = np.full((len(searchspace_val_data.index)), False)

            # Repeat the top 20% subgroup analysis on the validation data
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
                    preds_top_subgroups = val_outcome_preds[targ][bool_vec]
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

                    ROC_tuple_20_val = roc_curve(validation_outcome_true_vals[targ][bool_vec], val_outcome_preds[targ][bool_vec])
                    val_AUROC_20 = auc(ROC_tuple_20_val[0], ROC_tuple_20_val[1])
                    PR_tuple_20_val = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec], val_outcome_preds[targ][bool_vec])
                    val_AUPRC_20 = auc(PR_tuple_20_val[1], PR_tuple_20_val[0])
                    #
                    rand_val_AUROC_20 = roc_auc_score(validation_outcome_true_vals[targ][bool_vec], rand_val_pred[bool_vec])
                    rand_precision, rand_recall, rand_thresholds = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec], rand_val_pred[bool_vec])
                    rand_val_AUPRC_20 = auc(rand_recall, rand_precision)
                    #
                    #
                    temp_auroc = []
                    temp_auprc = []
                    for i in range(len(many_outcome_preds.columns)):
                        precision, recall, thresholds = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec], many_val_outcome_preds.iloc[:,i][bool_vec])
                        temp_auprc.append(auc(recall, precision))
                        temp_auroc.append(roc_auc_score(validation_outcome_true_vals[targ][bool_vec], many_val_outcome_preds.iloc[:,i][bool_vec]))
                    #
                    val_AUROC_20_sd = np.std(temp_auroc, ddof=1)
                    val_AUROC_20_mean = np.mean(temp_auroc)
                    val_AUPRC_20_sd = np.std(temp_auprc, ddof=1)
                    val_AUPRC_20_mean = np.mean(temp_auprc)

                    # Save predictions in top subgroups over iters
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
                #
            #
            iter_results[targ+pred_type] = [kfold_AUROC_mean, kfold_AUROC_sd, val_AUROC_mean, val_AUROC_sd, kfold_AUROC_20_mean, kfold_AUROC_20_sd, val_AUROC_20_mean, val_AUROC_20_sd,
            kfold_AUPRC_mean, kfold_AUPRC_sd, val_AUPRC_mean, val_AUPRC_sd, kfold_AUPRC_20_mean, kfold_AUPRC_20_sd, val_AUPRC_20_mean, val_AUPRC_20_sd]
            #
            all_results[targ+pred_type] = [subgroup_results_df, subgroup_val_results_df, kfold_AUROC, kfold_AUPRC, val_AUROC, val_AUPRC, PR_tuple_20, PR_tuple_20_val, kfold_AUPRC_20, val_AUPRC_20, ROC_tuple_20, ROC_tuple_20_val, kfold_AUROC_20, val_AUROC_20, rand_AUROC, rand_AUPRC, rand_val_AUROC, rand_val_AUPRC, rand_AUROC_20, rand_AUPRC_20, rand_val_AUROC_20, rand_val_AUPRC_20]
            #
        #
        #
        #save to file
        with open(output_dir + metric + "_prediction_results.pkl", "wb") as f:
            pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
        #
        print("saving results to .pkl")
        #
        #
        ####################################################################
        ## Save results + recompile into .csv's 4 R visualization scripts ##
        ####################################################################
        #
        #initialize writer
        workbook = xlsxwriter.Workbook(output_dir + metric + "_prediction_results.xlsx")
        worksheet_baseline = workbook.add_worksheet("baseline")
        worksheet_baseline_mean = workbook.add_worksheet("Mean+SD Across Preds")
        worksheet_baseline_rand = workbook.add_worksheet("rand baseline")
        worksheet_20 = workbook.add_worksheet("baseline @ 20% Data")
        worksheet_20_mean = workbook.add_worksheet("Mean+SD Across Preds @ 20% Data")
        worksheet_20_rand = workbook.add_worksheet("rand baseline @ 20% Data")
        #
        for outcome in outcome_order:
            targ=outcome+"_any"
            pred_type = "_tp"
            #
            worksheet_train = workbook.add_worksheet(outcome+"-train")
            worksheet_val = workbook.add_worksheet(outcome+"-val")
            worksheet_pr = workbook.add_worksheet(outcome+"-Kfold PR @ 20%")
            worksheet_roc = workbook.add_worksheet(outcome+"-Kfold ROC @ 20%")
            worksheet_pr_val = workbook.add_worksheet(outcome+"-Val PR @ 20%")
            worksheet_roc_val = workbook.add_worksheet(outcome+"-Val ROC @ 20%")
            #
            subgroup_results = all_results[targ+pred_type][0]
            subgroup_val_results = all_results[targ+pred_type][1]
            #
            kfold_AUROC = all_results[targ+pred_type][2]
            kfold_AUPRC = all_results[targ+pred_type][3]
            #
            val_AUROC = all_results[targ+pred_type][4]
            val_AUPRC = all_results[targ+pred_type][5]
            #
            (precision_20, recall_20, thresholds_20) = all_results[targ+pred_type][6]
            (precision_val_20, recall_val_20, thresholds_val_20) = all_results[targ+pred_type][7]
            #
            kfold_20_AUPRC = all_results[targ+pred_type][8]
            val_20_AUPRC = all_results[targ+pred_type][9]
            #
            (fpr_20, tpr_20, thresholds_20) = all_results[targ+pred_type][10]
            (fpr_val_20, tpr_val_20, thresholds_val_20) = all_results[targ+pred_type][11]
            #
            kfold_20_AUROC = all_results[targ+pred_type][12]
            val_20_AUROC = all_results[targ+pred_type][13]
            #
            #
            worksheet_baseline.write(0,0, "outcome")
            worksheet_baseline.write(0,1, "kfold AUROC")
            worksheet_baseline.write(0,2, "kfold AUPRC")
            worksheet_baseline.write(0,3, "val AUROC")
            worksheet_baseline.write(0,4, "val AUPRC")
            #
            worksheet_baseline.write(outcome_order.index(outcome)+1, 0, outcome)
            worksheet_baseline.write(outcome_order.index(outcome)+1, 1, kfold_AUROC)
            worksheet_baseline.write(outcome_order.index(outcome)+1, 2, kfold_AUPRC)
            worksheet_baseline.write(outcome_order.index(outcome)+1, 3, val_AUROC)
            worksheet_baseline.write(outcome_order.index(outcome)+1, 4, val_AUPRC)
            #
            #
            worksheet_20.write(0,0, "outcome")
            worksheet_20.write(0,1, "kfold AUPRC")
            worksheet_20.write(0,2, "val AUPRC")
            worksheet_20.write(0,3, "kfold AUROC")
            worksheet_20.write(0,4, "val AUROC")
            #
            worksheet_20.write(outcome_order.index(outcome)+1, 0, outcome)
            worksheet_20.write(outcome_order.index(outcome)+1, 1, kfold_20_AUPRC)
            worksheet_20.write(outcome_order.index(outcome)+1, 2, val_20_AUPRC)
            worksheet_20.write(outcome_order.index(outcome)+1, 3, kfold_20_AUROC)
            worksheet_20.write(outcome_order.index(outcome)+1, 4, val_20_AUROC)
            #
            #
            kfold_AUROC_mean = iter_results[targ+pred_type][0]
            kfold_AUROC_sd = iter_results[targ+pred_type][1]
            val_AUROC_mean = iter_results[targ+pred_type][2]
            val_AUROC_sd = iter_results[targ+pred_type][3]
            #
            kfold_AUROC_mean_20 = iter_results[targ+pred_type][4]
            kfold_AUROC_sd_20 = iter_results[targ+pred_type][5]
            val_AUROC_mean_20 = iter_results[targ+pred_type][6]
            val_AUROC_sd_20 = iter_results[targ+pred_type][7]
            #
            kfold_AUPRC_mean = iter_results[targ+pred_type][8]
            kfold_AUPRC_sd = iter_results[targ+pred_type][9]
            val_AUPRC_mean = iter_results[targ+pred_type][10]
            val_AUPRC_sd = iter_results[targ+pred_type][11]
            #
            kfold_AUPRC_mean_20 = iter_results[targ+pred_type][12]
            kfold_AUPRC_sd_20 = iter_results[targ+pred_type][13]
            val_AUPRC_mean_20 = iter_results[targ+pred_type][14]
            val_AUPRC_sd_20 = iter_results[targ+pred_type][15]
            #
            worksheet_20_mean.write(0,0, "outcome")
            worksheet_20_mean.write(0,1, "kfold AUPRC")
            worksheet_20_mean.write(0,2, "val AUPRC")
            worksheet_20_mean.write(0,3, "kfold AUROC")
            worksheet_20_mean.write(0,4, "val AUROC")
            #
            worksheet_20_mean.write(0,6, "kfold AUPRC SD")
            worksheet_20_mean.write(0,7, "val AUPRC SD")
            worksheet_20_mean.write(0,8, "kfold AUROC SD")
            worksheet_20_mean.write(0,9, "val AUROC SD")
            #
            #
            worksheet_20_mean.write(outcome_order.index(outcome)+1, 0, outcome)
            worksheet_20_mean.write(outcome_order.index(outcome)+1, 1, kfold_AUPRC_mean_20)
            worksheet_20_mean.write(outcome_order.index(outcome)+1, 2, val_AUPRC_mean_20)
            worksheet_20_mean.write(outcome_order.index(outcome)+1, 3, kfold_AUROC_mean_20)
            worksheet_20_mean.write(outcome_order.index(outcome)+1, 4, val_AUROC_mean_20)
            #
            worksheet_20_mean.write(outcome_order.index(outcome)+1, 6, kfold_AUPRC_sd_20)
            worksheet_20_mean.write(outcome_order.index(outcome)+1, 7, val_AUPRC_sd_20)
            worksheet_20_mean.write(outcome_order.index(outcome)+1, 8, kfold_AUROC_sd_20)
            worksheet_20_mean.write(outcome_order.index(outcome)+1, 9, val_AUROC_sd_20)
            #
            worksheet_baseline_mean.write(0,0, "outcome")
            worksheet_baseline_mean.write(0,1, "kfold AUROC")
            worksheet_baseline_mean.write(0,2, "kfold AUPRC")
            worksheet_baseline_mean.write(0,3, "val AUROC")
            worksheet_baseline_mean.write(0,4, "val AUPRC")
            #
            worksheet_baseline_mean.write(0,6, "kfold AUPRC SD")
            worksheet_baseline_mean.write(0,7, "val AUPRC SD")
            worksheet_baseline_mean.write(0,8, "kfold AUROC SD")
            worksheet_baseline_mean.write(0,9, "val AUROC SD")
            #
            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 0, outcome)
            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 1, kfold_AUROC_mean)
            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 2, kfold_AUPRC_mean)
            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 3, val_AUROC_mean)
            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 4, val_AUPRC_mean)
            #
            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 6, kfold_AUPRC_sd)
            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 7, val_AUPRC_sd)
            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 8, kfold_AUROC_sd)
            worksheet_baseline_mean.write(outcome_order.index(outcome)+1, 9, val_AUROC_sd)
            #
            #
            rand_AUROC = all_results[targ+pred_type][14]
            rand_AUPRC = all_results[targ+pred_type][15]
            #
            rand_val_AUROC = all_results[targ+pred_type][16]
            rand_val_AUPRC = all_results[targ+pred_type][17]
            #
            #
            rand_AUROC_20 = all_results[targ+pred_type][18]
            rand_AUPRC_20 = all_results[targ+pred_type][19]
            #
            rand_val_AUROC_20 = all_results[targ+pred_type][20]
            rand_val_AUPRC_20 = all_results[targ+pred_type][21]
            #
            worksheet_20_rand.write(0,0, "outcome")
            worksheet_20_rand.write(0,1, "kfold AUPRC")
            worksheet_20_rand.write(0,2, "val AUPRC")
            worksheet_20_rand.write(0,3, "kfold AUROC")
            worksheet_20_rand.write(0,4, "val AUROC")
            #
            worksheet_20_rand.write(outcome_order.index(outcome)+1, 0, outcome)
            worksheet_20_rand.write(outcome_order.index(outcome)+1, 1, rand_AUPRC_20)
            worksheet_20_rand.write(outcome_order.index(outcome)+1, 2, rand_val_AUPRC_20)
            worksheet_20_rand.write(outcome_order.index(outcome)+1, 3, rand_AUROC_20)
            worksheet_20_rand.write(outcome_order.index(outcome)+1, 4, rand_val_AUROC_20)
            #
            worksheet_baseline_rand.write(0,0, "outcome")
            worksheet_baseline_rand.write(0,1, "kfold AUROC")
            worksheet_baseline_rand.write(0,2, "kfold AUPRC")
            worksheet_baseline_rand.write(0,3, "val AUROC")
            worksheet_baseline_rand.write(0,4, "val AUPRC")
            #
            worksheet_baseline_rand.write(outcome_order.index(outcome)+1, 0, outcome)
            worksheet_baseline_rand.write(outcome_order.index(outcome)+1, 1, rand_AUROC)
            worksheet_baseline_rand.write(outcome_order.index(outcome)+1, 2, rand_AUPRC)
            worksheet_baseline_rand.write(outcome_order.index(outcome)+1, 3, rand_val_AUROC)
            worksheet_baseline_rand.write(outcome_order.index(outcome)+1, 4, rand_val_AUPRC)
            #
            #adding column titles
            col_num=0
            row_num=0
            for col in subgroup_results.columns:
                temp = worksheet_train.write(row_num, col_num, col)
                col_num = col_num + 1
            #
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
            #
            #adding column titles
            col_num=0
            row_num=0
            for col in subgroup_val_results.columns:
                temp = worksheet_val.write(row_num, col_num, col)
                col_num = col_num + 1
            #
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
            #
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
            #
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
        #
        workbook.close()
        print("writing results to .csv")

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
