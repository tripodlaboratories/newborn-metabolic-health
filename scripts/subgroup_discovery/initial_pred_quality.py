######################################################################################
## Script generates initial prediction AUROC/AUPRC and optimal prediction threshold ##
######################################################################################

#notes on the following scripts which may be of importance
#NOTE: all thresholds are determined by k-fold "training" data

# library imports
# general imports
import pandas as pd
import numpy as np

# import of model evaluation metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

#setting printing parameters so subgroup descriptions actually display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 250)
pd.set_option('display.max_colwidth', 150)

#Defining the following script as a function to maintain proper variable scoping
#as variable names are used across multiple scripts

def getInitResults():

    ######################################################################
    ## Read in of preds, and true values to determine model performance ##
    ######################################################################

    # read in predicted values
    results_dir = "./results/deep_mtl/neonatal/validation/ensemble/"
    preds = pd.read_csv(results_dir + "preds.csv")
    val_preds = pd.read_csv(results_dir + "valid_preds.csv")
    true_preds = pd.read_csv(results_dir + "true_vals.csv")
    true_val_preds= pd.read_csv("./data/processed/neonatal_conditions.csv", low_memory=False)

    #remane pred columns to be consistent
    val_preds = val_preds.rename(columns={"Unnamed: 0":"row_id"})
    preds = preds.rename(columns={"Unnamed: 0":"row_id"})

    #average over all iteration runs, previously was only taking one
    preds = preds.groupby(["row_id"])["nec_any","rop_any","bpd_any", "ivh_any"].mean() #smart way (it is, double checked against stupid for loop method)
    val_preds = val_preds.groupby(["row_id"])["nec_any","rop_any","bpd_any", "ivh_any"].mean() #smart way (it is, double checked against stupid for loop method)
    true_preds = true_preds.groupby(["row_id"])["nec_any","rop_any","bpd_any", "ivh_any"].mean() #smart way (it is, double checked against stupid for loop method)
    true_val_preds = true_val_preds.groupby(["row_id"])["nec_any","rop_any","bpd_any", "ivh_any"].mean() #smart way (it is, double checked against stupid for loop method)

    #reduce true values to those which were predicted for validation
    true_val_preds = true_val_preds.loc[val_preds.index]

    ###############################
    ## Align all data structures ##
    ###############################

    #sanity check that all indices are the same
    #(preds.index == true_preds.index).all()
    #(val_preds.index == true_val_preds.index).all()

    ########################################################################
    ## manipulate predicted probs into class labels via different cutoffs ##
    ########################################################################

    #this will create the target matrix, possible targets include true classification, true positivies, ect (based on a CUTOFF value)

    #from preds/actual values construct different classifications for good/bad prediction
    class_pred = preds[["nec_any","rop_any","bpd_any","ivh_any"]].transform(lambda z: z.transform(lambda x: 1 if x >= 0.5 else 0))
    class_val_pred = val_preds[["nec_any","rop_any","bpd_any","ivh_any"]].transform(lambda z: z.transform(lambda x: 1 if x >= 0.5 else 0))

    mis_class = {}
    for col in class_pred.columns:
        if(col != "row_id"):
            mis_class[col] = class_pred[col] == true_preds[col]

    #realign list into pandas dataframe
    mis_class = pd.DataFrame(mis_class)

    mis_val_class = {}
    for col in class_val_pred.columns:
        if(col != "row_id"):
            mis_val_class[col] = class_val_pred[col] == true_val_preds[col]

    #realign list into pandas dataframe
    mis_val_class = pd.DataFrame(mis_val_class)

    ######################################################
    ## initial performance with cutoff set at naive 0.5 ##
    ######################################################

    initial_performance = []
    for elem in ["nec_any","rop_any","bpd_any","ivh_any"]:
        #limiting to TRUE healthy controls (removing controls with positive co-outcomes)
        keep = (true_preds[np.setdiff1d(["nec_any","rop_any","bpd_any","ivh_any"], [elem])].sum(axis=1) == 0) | (true_preds[elem] == 1)
        cm = confusion_matrix(true_preds[elem][keep], class_pred[elem][keep])
        #compile confusion matrix results
        p = cm.sum(axis=1)[1]
        n = cm.sum(axis=1)[0]
        tn = cm[0,0]
        fn = cm[1,0]
        fp = cm[0,1]
        tp = cm[1,1]
        #compile metrics of performance
        preci = tp/(tp+fp)
        recal = tp/(tp+fn)
        f1 = 2*(preci * recal)/(preci+recal)
        AUROC = roc_auc_score(true_preds[elem][keep], preds[elem][keep])
        precision, recall, thresholds = precision_recall_curve(true_preds[elem][keep], preds[elem][keep])
        AUPRC = auc(recall, precision)
        fscore = (2 * precision * recall) / (precision + recall)
        fscore_cutoff = np.nanargmax(fscore)
        hyp_lift = 1/(mis_class[elem].sum()/len(mis_class[elem]))
        #append all data to array
        initial_performance.append([elem, tp,fp,fn,tn,preci,recal,f1,hyp_lift,thresholds[fscore_cutoff],AUPRC,AUROC])

    init_results_df = pd.DataFrame(initial_performance, columns=["response","TP","FP","FN","TN", "precision","recall", "F1", "max hypothetical Lift","F-score Cutoff","AUPRC", "AUROC"])

    #calculating performance on validation data
    initial_val_performance = []
    for elem in ["nec_any","rop_any","bpd_any","ivh_any"]:
    #limiting to TRUE healthy controls (removing controls with positive co-outcomes)
        keep = (true_val_preds[np.setdiff1d(["nec_any","rop_any","bpd_any","ivh_any"], [elem])].sum(axis=1) == 0) | (true_val_preds[elem] == 1)
        cm = confusion_matrix(true_val_preds[elem][keep], class_val_pred[elem][keep])
        #compile confusion matrix results
        p = cm.sum(axis=1)[1]
        n = cm.sum(axis=1)[0]
        tn = cm[0,0]
        fn = cm[1,0]
        fp = cm[0,1]
        tp = cm[1,1]
        #compile metrics of performance
        preci = tp/(tp+fp)
        recal = tp/(tp+fn)
        f1 = 2*(preci * recal)/(preci+recal)
        AUROC = roc_auc_score(true_val_preds[elem][keep], val_preds[elem][keep])
        precision, recall, thresholds = precision_recall_curve(true_val_preds[elem][keep], val_preds[elem][keep])
        AUPRC = auc(recall, precision)
        fscore = (2 * precision * recall) / (precision + recall)
        fscore_cutoff = np.nanargmax(fscore)
        hyp_lift = 1/(mis_val_class[elem].sum()/len(mis_val_class[elem]))
        #append all data to array
        initial_val_performance.append([elem, tp,fp,fn,tn,preci,recal,f1,hyp_lift,thresholds[fscore_cutoff],AUPRC,AUROC])

    init_val_results_df = pd.DataFrame(initial_val_performance, columns=["response","TP","FP","FN","TN", "precision","recall", "F1", "max hypothetical Lift","F-score Cutoff","AUPRC", "AUROC"])

    #############################################################
    ## initial performance with cutoff set to maximize f-score ##
    #############################################################

    initial_performance = []
    for elem in ["nec_any","rop_any","bpd_any","ivh_any", "nec_any"]:
        cutoff = init_results_df[init_results_df["response"] == elem]["F-score Cutoff"].iloc[0]
        class_pred = preds[["nec_any","rop_any","bpd_any","ivh_any"]].transform(lambda z: z.transform(lambda x: 1 if x >= cutoff else 0))
        keep = (true_preds[np.setdiff1d(["nec_any","rop_any","bpd_any","ivh_any"], [elem])].sum(axis=1) == 0) | (true_preds[elem] == 1)
        cm = confusion_matrix(true_preds[elem][keep], class_pred[elem][keep])
        #compile confusion matrix results
        p = cm.sum(axis=1)[1]
        n = cm.sum(axis=1)[0]
        tn = cm[0,0]
        fn = cm[0,1]
        fp = cm[1,0]
        tp = cm[1,1]
        #compile metrics of performance
        preci = tp/(tp+fp)
        recal = tp/(tp+fn)
        f1 = 2*(preci * recal)/(preci+recal)
        AUROC = roc_auc_score(true_preds[elem][keep], preds[elem][keep])
        precision, recall, thresholds = precision_recall_curve(true_preds[elem][keep], preds[elem][keep])
        AUPRC = auc(recall, precision)
        fscore = (2 * precision * recall) / (precision + recall)
        fscore_cutoff = np.nanargmax(fscore)
        hyp_lift = 1/(mis_class[elem].sum()/len(mis_class[elem]))
        #append all data to array
        initial_performance.append([elem, tp,fp,fn,tn,preci,recal,f1,hyp_lift,thresholds[fscore_cutoff],AUPRC,AUROC])

    init_results_df2 = pd.DataFrame(initial_performance, columns=["response","TP","FP","FN","TN", "precision","recall", "F1", "max hypothetical Lift","F-score Cutoff","AUPRC", "AUROC"])

    return init_results_df

