###############################################################################
## Code snippet to calc mean AUPRC from iter preds for bottleneck Prediction ##
###############################################################################

# general imports
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

#setting printing parameters so subgroup descriptions actually display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 250)
pd.set_option('display.max_colwidth', 150)

###################################################################################
## Read in of preds, true values, and data used for subgroup disc (metadata.csv) ##
###################################################################################

results_dir = "./results/deep_mtl/neonatal_bottleneck_validation/"
exp = "ensemble_bottle_1"

preds = pd.read_csv(results_dir + exp + "/bottleneck.csv")
val_preds = pd.read_csv(results_dir + exp + "/valid_bottleneck.csv")
true_preds = pd.read_csv(results_dir + exp + "/true_vals.csv")

true_val_preds= pd.read_csv("./data/processed/neonatal_conditions.csv", low_memory=False)[["nec_any","rop_any","bpd_any", "ivh_any", "row_id"]]

#remane pred columns to be consistent
val_preds = val_preds.rename(columns={"Unnamed: 0":"row_id"})
preds = preds.rename(columns={"Unnamed: 0":"row_id"})

#maintain predictions across individual iterations
many_preds = preds.copy().pivot_table(index="row_id",columns="iter",values="bottleneck_unit_0")
many_val_preds = val_preds.copy().pivot_table(index="row_id",columns="iter",values="bottleneck_unit_0")

#average over all iteration runs, previously was only taking one
preds = preds.groupby(["row_id"])["bottleneck_unit_0"].mean()
val_preds = val_preds.groupby(["row_id"])["bottleneck_unit_0"].mean()
true_preds = true_preds.groupby(["row_id"])["nec_any","rop_any","bpd_any", "ivh_any"].mean()

#collapse all outcomes to patients x outcomes dataframe while subgsetting to only obs in heldout validation
true_val_preds = true_val_preds.pivot_table(index="row_id").loc[val_preds.index]

#Since we are interested in identifying HEALTHY individuals target for prediction will be switched to healthy obs
true_preds = 1 - true_preds
true_val_preds = 1 - true_val_preds

outcome_order = ["bpd", "rop", "ivh", "nec"]
for outcome in outcome_order:
    print("starting analysis of - " + outcome)
    #this will create the target matrix, possible targets include true classification, true positivies, ect (based on a CUTOFF value)
    targ=outcome+"_any"
    pred_type = "_tp"
    #
    #limiting to TRUE healthy controls (removing controls with positive co-outcomes)
    keep = (true_preds[np.setdiff1d(["nec_any","rop_any","bpd_any","ivh_any"], [targ])].sum(axis=1) == 3) | (true_preds[targ] == 0)
    keep_val = (true_val_preds[np.setdiff1d(["nec_any","rop_any","bpd_any","ivh_any"], [targ])].sum(axis=1) == 3) | (true_val_preds[targ] == 0)
    #
    #k-fold
    preds_outcome = preds.loc[keep]
    true_preds_outcome = true_preds.loc[keep,:]
    many_preds_outcome = many_preds.loc[keep]
    #
    #validation
    val_preds_outcome = val_preds.loc[keep_val]
    true_val_preds_outcome = true_val_preds.loc[keep_val,:]
    many_val_preds_outcome = many_val_preds.loc[keep_val]
    #
    #
    temp_auprc = []
    for i in range(len(many_preds.columns)):
        precision1, recall1, thresholds = precision_recall_curve(true_val_preds_outcome[targ], many_val_preds_outcome.iloc[:,i])
        temp_auprc.append(auc(recall1, precision1))
    #
    val_AUPRC_sd = np.std(temp_auprc, ddof=1)
    val_AUPRC_mean = np.mean(temp_auprc)
    #
    temp_auprc = []
    for i in range(len(many_preds.columns)):
        precision, recall, thresholds = precision_recall_curve(true_preds_outcome[targ], many_preds_outcome.iloc[:,i])
        temp_auprc.append(auc(recall, precision))
    #
    kfold_AUPRC_sd = np.std(temp_auprc, ddof=1)
    kfold_AUPRC_mean = np.mean(temp_auprc)
    #
    precision, recall, thresholds = precision_recall_curve(true_preds_outcome[targ], preds_outcome)
    kfold_AUPRC = auc(recall, precision)
    #
    precision, recall, thresholds = precision_recall_curve(true_val_preds_outcome[targ], val_preds_outcome)
    val_AUPRC = auc(recall, precision)
    #
    print("")
    print("")
    print("---------------------"+outcome+"-----------------------")
    print("kfold AUPRC: "+str(np.round(kfold_AUPRC,3)))
    print("kfold mean AUPRC: "+str(np.round(kfold_AUPRC_mean,3)))
    print("kfold sd AUPRC: "+str(np.round(kfold_AUPRC_sd,3)))
    print("-----------------------------------------------")
    print("val AUPRC: "+str(np.round(val_AUPRC,3)))
    print("val mean AUPRC: "+str(np.round(val_AUPRC_mean,3)))
    print("val sd AUPRC: "+str(np.round(val_AUPRC_sd,3)))
    print("-----------------------------------------------")


###############################
## End of Bottleneck Results ##
###############################

###############################################################################
## Code snippet to calc mean AUPRC from iter preds for bottleneck Prediction ##
###############################################################################
results_dir = "./results/deep_mtl/neonatal/validation/"

# read in previous predictions
preds = pd.read_csv(results_dir + "preds.csv")
val_preds = pd.read_csv(results_dir + "valid_preds.csv")
true_preds = pd.read_csv(results_dir + "true_vals.csv")

true_val_preds= pd.read_csv("./data/processed/neonatal_conditions.csv", low_memory=False)


#remane pred columns to be consistent
val_preds = val_preds.rename(columns={"Unnamed: 0":"row_id"})
preds = preds.rename(columns={"Unnamed: 0":"row_id"})

#maintain predictions across individual model runs so that SD can be calculated
#also re-organizing so it is a row_id X run matrix
many_preds = {x: preds.copy().pivot_table(index="row_id",columns="iter", values=x+"_any") for x in outcome_order}
many_val_preds = {x: val_preds.copy().pivot_table(index="row_id",columns="iter", values=x+"_any") for x in outcome_order}

preds = preds.groupby(["row_id"])["nec_any","rop_any","bpd_any", "ivh_any"].mean()
val_preds = val_preds.groupby(["row_id"])["nec_any","rop_any","bpd_any", "ivh_any"].mean()
true_preds = true_preds.groupby(["row_id"])["nec_any","rop_any","bpd_any", "ivh_any"].mean()

#collapse all outcomes to patients x outcomes dataframe
true_val_preds = true_val_preds.groupby(["row_id"])["nec_any","rop_any","bpd_any", "ivh_any"].mean() #smart way (it is, double checked against stupid for loop method)
true_val_preds = true_val_preds.loc[val_preds.index]


#start of loop to collect all data
for outcome in outcome_order:
    print("starting analysis of - " + outcome)
    #this will create the target matrix, possible targets include true classification, true positivies, ect (based on a CUTOFF value)
    targ=outcome+"_any"
    pred_type = "_tp"
    #
    #limiting to TRUE healthy controls (removing controls with positive co-outcomes)
    keep = (true_preds[np.setdiff1d(["nec_any","rop_any","bpd_any","ivh_any"], [targ])].sum(axis=1) == 0) | (true_preds[targ] == 1)
    keep_val = (true_val_preds[np.setdiff1d(["nec_any","rop_any","bpd_any","ivh_any"], [targ])].sum(axis=1) == 0) | (true_val_preds[targ] == 1)
    #
    #k-fold
    preds_outcome = preds.loc[keep,:]
    true_preds_outcome = true_preds.loc[keep,:]
    #
    #validation
    val_preds_outcome = val_preds.loc[keep_val,:]
    true_val_preds_outcome = true_val_preds.loc[keep_val,:]
    #
    many_preds_outcome = many_preds[outcome].loc[keep]
    many_val_preds_outcome = many_val_preds[outcome].loc[keep_val]
    #
    temp_auprc = []
    for i in range(len(many_preds_outcome.columns)):
        precision1, recall1, thresholds = precision_recall_curve(true_val_preds_outcome[targ], many_val_preds_outcome.iloc[:,i])
        temp_auprc.append(auc(recall1, precision1))
    #
    val_AUPRC_sd = np.std(temp_auprc, ddof=1)
    val_AUPRC_mean = np.mean(temp_auprc)
    #
    #
    temp_auprc = []
    for i in range(len(many_preds_outcome.columns)):
        precision, recall, thresholds = precision_recall_curve(true_preds_outcome[targ], many_preds_outcome.iloc[:,i])
        temp_auprc.append(auc(recall, precision))
    #
    kfold_AUPRC_sd = np.std(temp_auprc, ddof=1)
    kfold_AUPRC_mean = np.mean(temp_auprc)
    #
    precision, recall, thresholds = precision_recall_curve(true_preds_outcome[targ], preds_outcome[targ])
    kfold_AUPRC = auc(recall, precision)
    #
    #
    precision, recall, thresholds = precision_recall_curve(true_val_preds_outcome[targ], val_preds_outcome[targ])
    val_AUPRC = auc(recall, precision)
    #
    print("")
    print("")
    print("---------------------"+outcome+"-----------------------")
    print("kfold AUPRC: "+str(np.round(kfold_AUPRC,3)))
    print("kfold mean AUPRC: "+str(np.round(kfold_AUPRC_mean,3)))
    print("kfold sd AUPRC: "+str(np.round(kfold_AUPRC_sd,3)))
    print("-----------------------------------------------")
    print("val AUPRC: "+str(np.round(val_AUPRC,3)))
    print("val mean AUPRC: "+str(np.round(val_AUPRC_mean,3)))
    print("val sd AUPRC: "+str(np.round(val_AUPRC_sd,3)))
    print("-----------------------------------------------")
