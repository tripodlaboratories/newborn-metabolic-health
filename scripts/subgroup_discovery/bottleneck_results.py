##########################################################################
## This script generates all subgroup results for bottleneck prediction ##
##########################################################################

# All library imports

# general imports
import pandas as pd
import numpy as np

# subgroup analysis package
import pysubgroup as ps

# sklearn imports for evaluating results
from sklearn.metrics import precision_recall_curve, roc_curve, auc #collects AUPRC + AUROC
from sklearn.metrics import confusion_matrix # used to collect true-pos. fals-negs, ect.

#importing some utilitie functions
import xlsxwriter
import pickle

#setting printing parameters so subgroup descriptions actually display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 250)
pd.set_option('display.max_colwidth', 150)

#Loads a function cutoffs which maximize F-score
exec(open("./scripts/subgroup_discovery/initial_bottleneck_unit_quality.py").read())

# this function returns cutoffs on K-fold Training data
init_results_df = getInitResults()
###################################################################################
## Read in of preds, true values, and data used for subgroup disc (metadata.csv) ##
###################################################################################

results_dir = "./results/deep_mtl/neonatal_bottleneck_validation/"
exp = "ensemble_bottle_1"

# read in alans predicted values
preds = pd.read_csv(results_dir+exp+"/bottleneck.csv")
val_preds = pd.read_csv(results_dir+exp+"/valid_bottleneck.csv")
true_preds = pd.read_csv(results_dir+exp+"/true_vals.csv")

# Read in raw data to get actual response for validation data, not currently included in prediction .csv's
true_val_preds= pd.read_csv("./data/processed/neonatal_conditions.csv", low_memory=False)

#remane pred columns to be consistent
val_preds = val_preds.rename(columns={"Unnamed: 0":"row_id"})
preds = preds.rename(columns={"Unnamed: 0":"row_id"})

#maintain predictions across individual model runs to calculate Mean + SD
many_preds = preds.copy().pivot_table(index="row_id",columns="iter",values="bottleneck_unit_0")
many_val_preds = val_preds.copy().pivot_table(index="row_id",columns="iter",values="bottleneck_unit_0")

#average over all iteration runs
preds = preds.groupby(["row_id"])["bottleneck_unit_0"].mean()
val_preds = val_preds.groupby(["row_id"])["bottleneck_unit_0"].mean()
true_preds = true_preds.groupby(["row_id"])["nec_any","rop_any","bpd_any", "ivh_any"].mean()

#collapse all outcomes to patients x outcomes dataframe
true_val_preds = true_val_preds.groupby(["row_id"])["nec_any","rop_any","bpd_any", "ivh_any"].mean() #smart way (it is, double checked against stupid for loop method)

metabolite_labels = pd.read_csv("./config/metabolite_labels.csv")
data = pd.read_csv("./data/processed/metadata.csv", low_memory=False)

#######################################################
## reduce *_data and *_values to only predicted data ##
#######################################################

#subset by observations
subset_data = data[data["gdspid"].isin(pd.unique(preds.index))]
subset_data = subset_data.set_index("gdspid")
subset_data = subset_data.drop("row_id", axis = 1)

#subset by observations
subset_val_data = data[data["gdspid"].isin(pd.unique(val_preds.index))]
subset_val_data = subset_val_data.set_index("gdspid")
subset_val_data = subset_val_data.drop("row_id", axis = 1)

#subsetting  holdout predictions
true_val_preds = true_val_preds.loc[subset_val_data.index]

###############################
## Align all data structures ##
###############################

#check that all indices are the same
(preds.index == subset_data.index).all()
(preds.index == true_preds.index).all()
(preds.index == many_preds.index).all()

(val_preds.index == subset_val_data.index).all()
(val_preds.index == true_val_preds.index).all()
(val_preds.index == many_val_preds.index).all()

########################################################################
## manipulate predicted probs into class labels via different cutoffs ##
########################################################################

#NOTE
#Since we are interested in identifying HEALTHY individuals
#target for prediction will be switched to healthy obs
true_preds = 1- true_preds
true_val_preds = 1- true_val_preds

outcome_order = ["bpd", "rop", "ivh", "nec"]
subgroup_alphas = {"bpd":0.15, "rop":0.325, "ivh":0.31, "nec":0.2}
subgroup_sizes = {"bpd":750, "rop":500, "ivh":300, "nec":300}

all_results = {}
iter_results = {}

# loop over neonatal outcomes to produce subgroup analysis results
for outcome in outcome_order:
    print("starting analysis of - " + outcome)
    #
    #this will create the target matrix, possible targets include true classification, true positivies, ect (based on a CUTOFF value)
    targ=outcome+"_any"
    pred_type = "_tp"
    #
    #limiting to TRUE healthy controls (removing controls for this outcome which have co-outcomes)
    keep = (true_preds[np.setdiff1d(["nec_any","rop_any","bpd_any","ivh_any"], [targ])].sum(axis=1) == 3) | (true_preds[targ] == 0)
    keep_val = (true_val_preds[np.setdiff1d(["nec_any","rop_any","bpd_any","ivh_any"], [targ])].sum(axis=1) == 3) | (true_val_preds[targ] == 0)
    #
    #k-fold
    preds_outcome = preds.loc[keep]
    true_preds_outcome = true_preds.loc[keep,:]
    subset_data_outcome = subset_data.loc[keep,:]
    #
    #validation
    val_preds_outcome = val_preds.loc[keep_val]
    true_val_preds_outcome = true_val_preds.loc[keep_val,:]
    subset_val_data_outcome = subset_val_data.loc[keep_val,:]
    #
    #each iter prediction to calculat SD
    many_preds_outcome = many_preds.loc[keep]
    many_val_preds_outcome = many_val_preds.loc[keep_val]
    #
    #double check that all indices are the same
    #(preds.index == subset_data.index).all()
    #(preds.index == true_preds.index).all()
    #(preds_outcome.index == many_preds_outcome.index).all()
    #
    #(val_preds.index == subset_val_data.index).all()
    #(val_preds.index == true_val_preds.index).all()
    #(val_preds_outcome.index == many_val_preds_outcome.index).all()
    #
    cutoff = init_results_df[init_results_df["response"] == targ]["F-score Cutoff"].iloc[0]
    #
    #based on f-score maximizing cutoff
    class_pred = preds_outcome.transform(lambda z: z.transform(lambda x: 1 if x >= cutoff else 0))
    class_val_pred = val_preds_outcome.transform(lambda z: z.transform(lambda x: 1 if x >= cutoff else 0))
    #
    #adapt for loop for additional subgroup discovery targets
    mis_class = {}
    for col in true_preds_outcome.columns:
        mis_class[col] = true_preds_outcome[col].astype("bool")
        mis_class[col+"_misclass"] = class_pred == true_preds_outcome[col]
        mis_class[col+"_tp"] = mis_class[col+"_misclass"] & (true_preds_outcome[col] == 1)
        mis_class[col+"_tn"] = mis_class[col+"_misclass"] & (true_preds_outcome[col] == 0)
    #
    subgroup_targets = pd.DataFrame(mis_class)
    #
    #adapt for loop for additional subgroup discovery targets
    mis_class = {}
    for col in true_val_preds_outcome.columns:
        mis_class[col] = true_val_preds_outcome[col].astype("bool")
        mis_class[col+"_misclass"] = class_val_pred == true_val_preds_outcome[col]
        mis_class[col+"_tp"] = mis_class[col+"_misclass"] & (true_val_preds_outcome[col] == 1)
        mis_class[col+"_tn"] = mis_class[col+"_misclass"] & (true_val_preds_outcome[col] == 0)
    #
    subgroup_val_targets = pd.DataFrame(mis_class)
    #
    ###########################################################################
    ## massage metabolites and demographic data into quantiles for discovery ##
    ###########################################################################
    #
    temp_data = subset_data_outcome[subset_data_outcome.columns.values[subset_data_outcome.isna().sum() == 0]].copy()
    temp_data[targ+pred_type] = subgroup_targets[targ+pred_type]
    #
    #constructing list of demographic and metabolomic features to keep
    keep_features = temp_data.columns.to_series().apply(lambda z: True if ("rc" in z) or (z in ["sex3", "mrace_catm", "payer_catm", "medu_catm", targ+pred_type]) else False)
    temp_data = temp_data[temp_data.columns[keep_features]]
    #
    #compile list of features which need to be transformed into quantiles
    transform = temp_data.apply(lambda z: z.name if (temp_data[z.name].dtype == "float64") or ("rc" in z.name) else None).unique()[1:]
    #np.setdiff1d(temp_data.columns,transform) #list columns that will not be transformed
    #
    searchspace_data = {}
    #
    #transform all data into various quantiles [2,3,5] as per martins experiment
    for c in temp_data.columns:
        if c in transform:
            for q in [2,3,5]:
                column = f"{c}_q-{q}"
                searchspace_data[column] = pd.qcut(temp_data[c], q, duplicates="drop").cat.codes
        else:
            searchspace_data[c] = temp_data[c]
    #
    searchspace_data = pd.DataFrame(searchspace_data)
    #
    # create indicator vector for metabolite columns
    is_metabolite = searchspace_data.columns.to_series().apply(lambda z: True if "rc" in z else False)
    #
    #########################################
    ## Massaging validation data similarly ##
    #########################################
    #
    temp_val_data = subset_val_data_outcome[subset_val_data_outcome.columns.values[subset_val_data_outcome.isna().sum() == 0]].copy()
    temp_val_data[targ+pred_type] = subgroup_val_targets[targ+pred_type]
    #
    #constructing list of demographic and metabolomic features to keep
    keep_val_features = temp_val_data.columns.to_series().apply(lambda z: True if ("rc" in z) or (z in ["sex3", "mrace_catm", "payer_catm", "medu_catm", targ+pred_type]) else False)
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
    searchspace_val_data = pd.DataFrame(searchspace_val_data)
    #
    # create indicator vector for metabolite columns
    is_val_metabolite = searchspace_val_data.columns.to_series().apply(lambda z: True if "rc" in z else False)
    #
    #########################################
    ## initial subgroup analysis of "targ" ##
    #########################################
    #
    #generating subgroup results on metabolites alone
    target = ps.BinaryTarget(targ+pred_type,True)
    searchspace = ps.create_selectors(searchspace_data[searchspace_data.columns[is_metabolite]], ignore=[targ+pred_type])
    task = ps.SubgroupDiscoveryTask(
        searchspace_data,
        target,
        searchspace,
        result_set_size=subgroup_sizes[outcome],
        depth=4, #controls the max number of features to combine
        qf=ps.StandardQF(a=subgroup_alphas[outcome]) #was 0.21
    )
    #
    results= ps.BeamSearch(beam_width=subgroup_sizes[outcome]).execute(task)
    #
    # iterate over and combine subgroups to extract information regarding subgroup predictive performance
    subgroup_desc = results.to_dataframe()["subgroup"]
    #
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
        #
        #collect general data for top 1:index(elem) subgroups
        precision, recall, thresholds = precision_recall_curve(true_preds_outcome[targ][bool_vec], preds_outcome[bool_vec])
        AUPRC = auc(recall, precision)
        precision, recall, thresholds = precision_recall_curve(true_preds_outcome[targ][bool_vec_inner], preds_outcome[bool_vec_inner])
        subgroup_AUPRC = auc(recall, precision)
        #
        #collect information regarding individual subgroup
        if(len(true_preds_outcome[targ][bool_vec]) == true_preds_outcome[targ][bool_vec].sum()):
            AUROC = np.nan
            AUROC_sd = np.nan
            AUROC_mean = np.nan
            #
            # Loops like this collect performance of individual model iterations
            temp_auprc = []
            for i in range(len(many_preds.columns)):
                precision, recall, thresholds = precision_recall_curve(true_preds_outcome[targ][bool_vec], many_preds_outcome.iloc[:,i][bool_vec])
                temp_auprc.append(auc(recall, precision))
            #
            AUPRC_sd = np.std(temp_auprc, ddof=1) #used to make std consistent with R
            AUPRC_mean = np.mean(temp_auprc)
            #
        else:
            AUROC = roc_auc_score(true_preds_outcome[targ][bool_vec], preds_outcome[bool_vec]) # #calculate sd metrics
            #
            temp_auroc = []
            temp_auprc = []
            for i in range(len(many_preds.columns)):
                precision, recall, thresholds = precision_recall_curve(true_preds_outcome[targ][bool_vec], many_preds_outcome.iloc[:,i][bool_vec])
                temp_auprc.append(auc(recall, precision))
                temp_auroc.append(roc_auc_score(true_preds_outcome[targ][bool_vec], many_preds_outcome.iloc[:,i][bool_vec])) #
            #
            AUROC_sd = np.std(temp_auroc, ddof=1)
            AUROC_mean = np.mean(temp_auroc)
            AUPRC_sd = np.std(temp_auprc, ddof=1)
            AUPRC_mean = np.mean(temp_auprc)
        #
        #
        if(len(true_preds_outcome[targ][bool_vec_inner]) == true_preds_outcome[targ][bool_vec_inner].sum()):
            subgroup_AUROC = np.nan
            subgroup_AUROC_sd = np.nan
            subgroup_AUROC_mean = np.nan
            #
            temp_auprc = []
            for i in range(len(many_preds.columns)):
                precision, recall, thresholds = precision_recall_curve(true_preds_outcome[targ][bool_vec_inner], many_preds_outcome.iloc[:,i][bool_vec_inner])
                temp_auprc.append(auc(recall, precision))
            #
            subgroup_AUPRC_sd = np.std(temp_auprc, ddof=1)
            subgroup_AUPRC_mean = np.mean(temp_auprc)
            #
        else:
            subgroup_AUROC = roc_auc_score(true_preds_outcome[targ][bool_vec_inner], preds_outcome[bool_vec_inner])
            #
            temp_auroc = []
            temp_auprc = []
            for i in range(len(many_preds.columns)):
                precision, recall, thresholds = precision_recall_curve(true_preds_outcome[targ][bool_vec_inner], many_preds_outcome.iloc[:,i][bool_vec_inner])
                temp_auprc.append(auc(recall, precision))
                temp_auroc.append(roc_auc_score(true_preds_outcome[targ][bool_vec_inner], many_preds_outcome.iloc[:,i][bool_vec_inner]))
            #
            subgroup_AUROC_sd = np.std(temp_auroc, ddof=1)
            subgroup_AUROC_mean = np.mean(temp_auroc)
            subgroup_AUPRC_sd = np.std(temp_auprc, ddof=1)
            subgroup_AUPRC_mean = np.mean(temp_auprc)
        #
        #collect results for this outcome to be found in 'subgroup_results_df'
        total_elem = elem + "-OR-" + total_elem
        data.append([total_elem, elem, bool_vec.sum(), bool_vec.sum()/len(searchspace_data[targ+pred_type]), bool_vec_inner.sum(), count, AUPRC, AUPRC_sd, subgroup_AUPRC, subgroup_AUPRC_sd, AUROC, AUROC_sd, subgroup_AUROC, subgroup_AUROC_sd])
    #
    subgroup_results_df = pd.DataFrame(data, columns=["total group", "subgroup", "size", "% data", "subgroup size", "num_groups", "AUPRC", "AUPRC SD","subgroup AUPRC","subgroup AUPRC SD", "AUROC", "AUROC SD", "subgroup AUROC", "subgroup AUROC SD"])
    #
    #
    kfold_AUROC = roc_auc_score(true_preds_outcome[targ], preds_outcome)
    precision, recall, thresholds = precision_recall_curve(true_preds_outcome[targ], preds_outcome)
    kfold_AUPRC = auc(recall, precision)
    #
    temp_auroc = []
    temp_auprc = []
    for i in range(len(many_preds.columns)):
        precision, recall, thresholds = precision_recall_curve(true_preds_outcome[targ], many_preds_outcome.iloc[:,i])
        temp_auprc.append(auc(recall, precision))
        temp_auroc.append(roc_auc_score(true_preds_outcome[targ], many_preds_outcome.iloc[:,i]))
    #
    kfold_AUROC_sd = np.std(temp_auroc, ddof=1)
    kfold_AUROC_mean = np.mean(temp_auroc)
    kfold_AUPRC_sd = np.std(temp_auprc, ddof=1)
    kfold_AUPRC_mean = np.mean(temp_auprc)
    #
    #########################################################################
    ## Collect AUROC, AUPRC, and other metrics on held out validation data ##
    #########################################################################
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
        precision, recall, thresholds = precision_recall_curve(true_val_preds_outcome[targ][bool_vec], val_preds_outcome[bool_vec])
        AUPRC = auc(recall, precision)
        precision, recall, thresholds = precision_recall_curve(true_val_preds_outcome[targ][bool_vec_inner], val_preds_outcome[bool_vec_inner])
        subgroup_AUPRC = auc(recall, precision)
        #
        #
        if(len(true_val_preds_outcome[targ][bool_vec]) == true_val_preds_outcome[targ][bool_vec].sum()):
            AUROC = np.nan
            AUROC_sd = np.nan
            AUROC_mean = np.nan
            #
            temp_auprc = []
            for i in range(len(many_preds.columns)):
                precision, recall, thresholds = precision_recall_curve(true_val_preds_outcome[targ][bool_vec], many_val_preds_outcome.iloc[:,i][bool_vec])
                temp_auprc.append(auc(recall, precision))
            #
            AUPRC_sd = np.std(temp_auprc, ddof=1)
            AUPRC_mean = np.mean(temp_auprc)
        #
        else:
            AUROC = roc_auc_score(true_val_preds_outcome[targ][bool_vec], val_preds_outcome[bool_vec])
            #
            temp_auroc = []
            temp_auprc = []
            for i in range(len(many_preds.columns)):
                precision, recall, thresholds = precision_recall_curve(true_val_preds_outcome[targ][bool_vec], many_val_preds_outcome.iloc[:,i][bool_vec])
                temp_auprc.append(auc(recall, precision))
                temp_auroc.append(roc_auc_score(true_val_preds_outcome[targ][bool_vec], many_val_preds_outcome.iloc[:,i][bool_vec]))
            #
            AUROC_sd = np.std(temp_auroc, ddof=1)
            AUROC_mean = np.mean(temp_auroc)
            AUPRC_sd = np.std(temp_auprc, ddof=1)
            AUPRC_mean = np.mean(temp_auprc)
        #
        #
        if(len(true_val_preds_outcome[targ][bool_vec_inner]) == true_val_preds_outcome[targ][bool_vec_inner].sum()):
            subgroup_AUROC = np.nan
            subgroup_AUROC_sd = np.nan
            subgroup_AUROC_mean = np.nan
            #
            temp_auprc = []
            for i in range(len(many_preds.columns)):
                precision, recall, thresholds = precision_recall_curve(true_val_preds_outcome[targ][bool_vec_inner], many_val_preds_outcome.iloc[:,i][bool_vec_inner])
                temp_auprc.append(auc(recall, precision))
            #
            subgroup_AUPRC_sd = np.std(temp_auprc, ddof=1)
            subgroup_AUPRC_mean = np.mean(temp_auprc)
        #
        else:
            subgroup_AUROC = roc_auc_score(true_val_preds_outcome[targ][bool_vec_inner], val_preds_outcome[bool_vec_inner])
            #
            #calculate sd metrics
            temp_auroc = []
            temp_auprc = []
            for i in range(len(many_preds.columns)):
                precision, recall, thresholds = precision_recall_curve(true_val_preds_outcome[targ][bool_vec_inner], many_val_preds_outcome.iloc[:,i][bool_vec_inner])
                temp_auprc.append(auc(recall, precision))
                temp_auroc.append(roc_auc_score(true_val_preds_outcome[targ][bool_vec_inner], many_val_preds_outcome.iloc[:,i][bool_vec_inner]))
            #
            subgroup_AUROC_sd = np.std(temp_auroc, ddof=1)
            subgroup_AUROC_mean = np.mean(temp_auroc)
            subgroup_AUPRC_sd = np.std(temp_auprc, ddof=1)
            subgroup_AUPRC_mean = np.mean(temp_auprc)
        #
        total_elem = elem + "-OR-" + total_elem
        data.append([total_elem, elem, bool_vec.sum(), bool_vec.sum()/len(searchspace_data[targ+pred_type]), bool_vec_inner.sum(), count, AUPRC, AUPRC_sd, subgroup_AUPRC, subgroup_AUPRC_sd, AUROC, AUROC_sd, subgroup_AUROC, subgroup_AUROC_sd])
    #
    subgroup_val_results_df = pd.DataFrame(data, columns=["total group", "subgroup", "size", "% data", "subgroup size", "num_groups", "AUPRC", "AUPRC SD","subgroup AUPRC","subgroup AUPRC SD", "AUROC", "AUROC SD", "subgroup AUROC", "subgroup AUROC SD"])
    #
    val_AUROC = roc_auc_score(true_val_preds_outcome[targ], val_preds_outcome)
    precision, recall, thresholds = precision_recall_curve(true_val_preds_outcome[targ], val_preds_outcome)
    val_AUPRC = auc(recall, precision)
    #
    temp_auroc = []
    temp_auprc = []
    for i in range(len(many_preds.columns)):
        precision1, recall1, thresholds = precision_recall_curve(true_val_preds_outcome[targ], many_val_preds_outcome.iloc[:,i])
        temp_auprc.append(auc(recall1, precision1))
        temp_auroc.append(roc_auc_score(true_val_preds_outcome[targ], many_val_preds_outcome.iloc[:,i]))
    #
    val_AUROC_sd = np.std(temp_auroc, ddof=1)
    val_AUROC_mean = np.mean(temp_auroc)
    val_AUPRC_sd = np.std(temp_auprc, ddof=1)
    val_AUPRC_mean = np.mean(temp_auprc)
    #
    #generate AUPRC from random vector
    np.random.seed(1234)
    rand_val_pred = np.random.uniform(0,1,len(true_val_preds_outcome[targ]))
    rand_pred = np.random.uniform(0,1,len(true_preds_outcome[targ]))
    #
    rand_AUROC = roc_auc_score(true_preds_outcome[targ], rand_pred)
    rand_precision, rand_recall, rand_thresholds = precision_recall_curve(true_preds_outcome[targ], rand_pred)
    rand_AUPRC = auc(rand_recall, rand_precision)
    #
    rand_val_AUROC = roc_auc_score(true_val_preds_outcome[targ], rand_val_pred)
    rand_precision, rand_recall, rand_thresholds = precision_recall_curve(true_val_preds_outcome[targ], rand_val_pred)
    rand_val_AUPRC = auc(rand_recall, rand_precision)
    #
    ##################################################################
    ## Collect precision/recall values for figure 3 PR-curve at 20% ##
    ##################################################################
    #
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
        if(count == select_index):
            ROC_tuple_20 = roc_curve(true_preds_outcome[targ][bool_vec], preds_outcome[bool_vec])
            kfold_AUROC_20 = auc(ROC_tuple_20[0], ROC_tuple_20[1])
            PR_tuple_20 = precision_recall_curve(true_preds_outcome[targ][bool_vec], preds_outcome[bool_vec])
            kfold_AUPRC_20 = auc(PR_tuple_20[1], PR_tuple_20[0])
            #
            rand_AUROC_20 = roc_auc_score(true_preds_outcome[targ][bool_vec], rand_pred[bool_vec])
            rand_precision, rand_recall, rand_thresholds = precision_recall_curve(true_preds_outcome[targ][bool_vec], rand_pred[bool_vec])
            rand_AUPRC_20 = auc(rand_recall, rand_precision)
            #
            temp_auroc = []
            temp_auprc = []
            for i in range(len(many_preds.columns)):
                precision, recall, thresholds = precision_recall_curve(true_preds_outcome[targ][bool_vec], many_preds_outcome.iloc[:,i][bool_vec])
                temp_auprc.append(auc(recall, precision))
                temp_auroc.append(roc_auc_score(true_preds_outcome[targ][bool_vec], many_preds_outcome.iloc[:,i][bool_vec]))
            #
            kfold_AUROC_20_sd = np.std(temp_auroc, ddof=1)
            kfold_AUROC_20_mean = np.mean(temp_auroc)
            kfold_AUPRC_20_sd = np.std(temp_auprc, ddof=1)
            kfold_AUPRC_20_mean = np.mean(temp_auprc)
        #
    #
    select =  (subgroup_val_results_df["% data"]* 100)
    select_index = select.index[select == min(select, key=lambda x:abs(x-20))][0]
    bool_vec = np.full((len(searchspace_val_data.index)), False)
    #
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
        if(count == select_index):
            ROC_tuple_20_val = roc_curve(true_val_preds_outcome[targ][bool_vec], val_preds_outcome[bool_vec])
            val_AUROC_20 = auc(ROC_tuple_20_val[0], ROC_tuple_20_val[1])
            PR_tuple_20_val = precision_recall_curve(true_val_preds_outcome[targ][bool_vec], val_preds_outcome[bool_vec])
            val_AUPRC_20 = auc(PR_tuple_20_val[1], PR_tuple_20_val[0])
            #
            rand_val_AUROC_20 = roc_auc_score(true_val_preds_outcome[targ][bool_vec], rand_val_pred[bool_vec])
            rand_precision, rand_recall, rand_thresholds = precision_recall_curve(true_val_preds_outcome[targ][bool_vec], rand_val_pred[bool_vec])
            rand_val_AUPRC_20 = auc(rand_recall, rand_precision)
            #
            temp_auroc = []
            temp_auprc = []
            for i in range(len(many_preds.columns)):
                precision, recall, thresholds = precision_recall_curve(true_val_preds_outcome[targ][bool_vec], many_val_preds_outcome.iloc[:,i][bool_vec])
                temp_auprc.append(auc(recall, precision))
                temp_auroc.append(roc_auc_score(true_val_preds_outcome[targ][bool_vec], many_val_preds_outcome.iloc[:,i][bool_vec]))
            #
            val_AUROC_20_sd = np.std(temp_auroc, ddof=1)
            val_AUROC_20_mean = np.mean(temp_auroc)
            val_AUPRC_20_sd = np.std(temp_auprc, ddof=1)
            val_AUPRC_20_mean = np.mean(temp_auprc)
    #
    #
    iter_results[targ+pred_type] = [kfold_AUROC_mean, kfold_AUROC_sd, val_AUROC_mean, val_AUROC_sd, kfold_AUROC_20_mean, kfold_AUROC_20_sd, val_AUROC_20_mean, val_AUROC_20_sd,
    kfold_AUPRC_mean, kfold_AUPRC_sd, val_AUPRC_mean, val_AUPRC_sd, kfold_AUPRC_20_mean, kfold_AUPRC_20_sd, val_AUPRC_20_mean, val_AUPRC_20_sd]
    #
    all_results[targ+pred_type] = [subgroup_results_df, subgroup_val_results_df, kfold_AUROC, kfold_AUPRC, val_AUROC, val_AUPRC, PR_tuple_20, PR_tuple_20_val, kfold_AUPRC_20, val_AUPRC_20, ROC_tuple_20, ROC_tuple_20_val, kfold_AUROC_20, val_AUROC_20, rand_AUROC, rand_AUPRC, rand_val_AUROC, rand_val_AUPRC, rand_AUROC_20, rand_AUPRC_20, rand_val_AUROC_20, rand_val_AUPRC_20, kfold_AUROC_sd, kfold_AUPRC_sd, val_AUROC_sd, val_AUPRC_sd, kfold_AUROC_20_sd, kfold_AUPRC_20_sd, val_AUROC_20_sd, val_AUPRC_20_sd]

#save to file
with open("./results/subgroup_discovery/subgroup_bottleneck_results.pkl", "wb") as f:
    pickle.dump([all_results, iter_results], f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

print("saving results to .pkl")

####################################################################
## Save results + recompile into .csv's 4 R visualization scripts ##
####################################################################

#initialize writer
workbook = xlsxwriter.Workbook("./results/subgroup_discovery/subgroup_bottleneck_results.xlsx")
worksheet_baseline = workbook.add_worksheet("baseline")
worksheet_baseline_mean = workbook.add_worksheet("Mean+SD Across Preds")
worksheet_baseline_rand = workbook.add_worksheet("rand baseline")
worksheet_20 = workbook.add_worksheet("baseline @ 20% Data")
worksheet_20_mean = workbook.add_worksheet("Mean+SD Across Preds @ 20% Data")
worksheet_20_rand = workbook.add_worksheet("rand baseline @ 20% Data")

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

workbook.close()
print("writing results to .csv")

