#################################################################################################
## Script generates initial bottleneck prediction AUROC/AUPRC and optimal bottleneck threshold ##
#################################################################################################

#notes on the following scripts which may be of importance
#NOTE: all thresholds are determined by k-fold "training" data
#NOTE: for this prediction task "healthy" is the target so predictions of neonatal
# outcomes are reversed ie true_prds = 1-true_preds

# library imports
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

    #################################################################################
    ## Read in of bottleneck_output and true values to determine model performance ##
    #################################################################################

    # read in predicted values
    results_dir = "./results/deep_mtl/neonatal_bottleneck_validation/ensemble_bottle_1/"
    bottleneck_output = pd.read_csv(results_dir + "bottleneck.csv")
    true_preds = pd.read_csv(results_dir + "true_vals.csv")

    bottleneck_val_output = pd.read_csv(results_dir + "valid_bottleneck.csv")
    true_val_preds= pd.read_csv("./data/processed/neonatal_conditions.csv", low_memory=False)

    #rename pred columns to be consistent
    bottleneck_output = bottleneck_output.rename(columns={"Unnamed: 0":"row_id"})
    bottleneck_val_output = bottleneck_val_output.rename(columns={"Unnamed: 0":"row_id"})

    #average over all iteration runs
    bottleneck_output = bottleneck_output.groupby(["row_id"])["bottleneck_unit_0"].mean() #smart way (it is, double checked against stupid for loop method)
    true_preds = true_preds.groupby(["row_id"])["nec_any","rop_any","bpd_any", "ivh_any"].mean()
    bottleneck_val_output = bottleneck_val_output.groupby(["row_id"])["bottleneck_unit_0"].mean()
    true_val_preds = true_val_preds.groupby(["row_id"])["nec_any","rop_any","bpd_any", "ivh_any"].mean()

    ########################################################################
    ## manipulate predicted probs into class labels via different cutoffs ##
    ########################################################################

    #this will create the target matrix, possible targets include true classification, true positivies, ect (based on a CUTOFF value)

    #from preds/actual values construct different classifications for good/bad prediction
    class_pred = bottleneck_output.transform(lambda z: z.transform(lambda x: 1 if x >= 0.5 else 0))

    mis_class = {}
    for col in true_preds.columns:
        if(col != "row_id"):
            mis_class[col] = class_pred == true_preds[col]

    #realign list into pandas dataframe
    mis_class = pd.DataFrame(mis_class)

    #from preds/actual values construct different classifications for good/bad prediction
    class_val_pred = bottleneck_val_output.transform(lambda z: z.transform(lambda x: 1 if x >= 0.5 else 0))

    mis_val_class = {}
    for col in true_val_preds.columns:
        if(col != "row_id"):
            mis_val_class[col] = class_val_pred == true_val_preds[col]

    #realign list into pandas dataframe
    mis_val_class = pd.DataFrame(mis_val_class)

    ######################################################
    ## initial performance with cutoff set at naive 0.5 ##
    ######################################################

    # this along with a change in the 'keep' vector is the only change from model predictions
    true_preds = 1- true_preds
    true_val_preds = 1- true_val_preds

    #sanity check for bottleneck index and preds index align
    (true_preds.index == class_pred.index).all()

    #loop over different outcomes and the bottleneck units ability to predict "healthy" vs
    #"non-healthy" for different neonatal outcomes
    initial_performance = []
    for elem in ["nec_any","rop_any","bpd_any","ivh_any"]:
        #limiting to TRUE healthy controls (removing controls with positive co-outcomes)
        keep = (true_preds[np.setdiff1d(["nec_any","rop_any","bpd_any","ivh_any"], [elem])].sum(axis=1) == 3) | (true_preds[elem] == 0)
        cm = confusion_matrix(true_preds[elem][keep], class_pred[keep])
        #compile confusion matrix results
        p = cm.sum(axis=1)[1]
        n = cm.sum(axis=1)[0]
        tn = cm[0,0]
        fn = cm[1,0]
        fp = cm[0,1]
        tp = cm[1,1]
        #compile AUC based metrics
        AUROC = roc_auc_score(true_preds[elem][keep], bottleneck_output[keep])
        precision, recall, thresholds = precision_recall_curve(true_preds[elem][keep], bottleneck_output[keep])
        AUPRC = auc(recall, precision)
        #compile standard metrics
        preci = tp/(tp+fp)
        recal = tp/(tp+fn)
        f1 = 2*(preci * recal)/(preci+recal)
        fscore = (2 * precision * recall) / (precision + recall)
        #NOTE: this is the cutoff value used for downstream subgroup analysis
        fscore_cutoff = np.nanargmax(fscore)
        hyp_lift = 1/(mis_class[elem].sum()/len(mis_class[elem]))
        #append all data to array
        initial_performance.append([elem, tp,fp,fn,tn, preci, recal, f1, hyp_lift, thresholds[fscore_cutoff], AUPRC, AUROC])

    #reconfigure results into dataframe
    init_results_df = pd.DataFrame(initial_performance, columns=["response","TP","FP","FN","TN", "precision","recall", "F1", "max hypothetical Lift","F-score Cutoff","AUPRC", "AUROC"])

    #############################################################
    ## initial performance with cutoff set to maximize f-score ##
    #############################################################

    initial_performance = []
    for elem in ["nec_any","rop_any","bpd_any","ivh_any"]:
        cutoff = init_results_df[init_results_df["response"] == elem]["F-score Cutoff"].iloc[0]
        class_pred = bottleneck_output.transform(lambda z: z.transform(lambda x: 1 if x >= cutoff else 0))
        keep = (true_preds[np.setdiff1d(["nec_any","rop_any","bpd_any","ivh_any"], [elem])].sum(axis=1) == 3) | (true_preds[elem] == 0)
        cm = confusion_matrix(true_preds[elem][keep], class_pred[keep])
        #compile confusion matrix results
        p = cm.sum(axis=1)[1]
        n = cm.sum(axis=1)[0]
        tn = cm[0,0]
        fn = cm[1,0]
        fp = cm[0,1]
        tp = cm[1,1]
        # compile AUC based metrics
        AUROC = roc_auc_score(true_preds[elem][keep], bottleneck_output[keep])
        precision, recall, thresholds = precision_recall_curve(true_preds[elem][keep], bottleneck_output[keep])
        AUPRC = auc(recall, precision)
        #compile standard metrics
        preci = tp/(tp+fp)
        recal = tp/(tp+fn)
        f1 = 2*(preci * recal)/(preci+recal)
        fscore = (2 * precision * recall) / (precision + recall)
        fscore_cutoff = np.nanargmax(fscore)
        hyp_lift = 1/(mis_class[elem].sum()/len(mis_class[elem]))
        #append all data to array
        initial_performance.append([elem, tp,fp,fn,tn,preci,recal,f1,hyp_lift,thresholds[fscore_cutoff],AUPRC,AUROC])

    init_results_df2 = pd.DataFrame(initial_performance, columns=["response","TP","FP","FN","TN", "precision","recall", "F1", "max hypothetical Lift","F-score Cutoff","AUPRC", "AUROC"])

    return init_results_df

