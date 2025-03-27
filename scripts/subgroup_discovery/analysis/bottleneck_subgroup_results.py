##########################################################################
## This script generates all subgroup results for bottleneck prediction ##
##########################################################################
# general imports
import argparse
import os
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
import yaml

#NOTE: altered pysubgroup package must be installed prior to analysis
#to install use the following command "pip install git+https://github.com/tripodlaboratories/pysubgroup-prediction.git@predictionQF"

import pysubgroup as ps
from biobank_project.subgroup_discovery import output

def get_args():
    parser = argparse.ArgumentParser(
        description='Perform subgroup discovery on bottleneck model outputs as health index',
        conflict_handler='resolve')
    parser.add_argument(
        '-i', '--input_directory', default='./results/deep_mtl/neonatal_bottleneck_validation/ensemble_bottle_1/',
        help='Input experiment directory, should contain bottleneck.csv, valid_bottleneck.csv, and true_vals.csv')
    parser.add_argument(
        '-o', '--output_directory',
        help='output directory to save results files')
    parser.add_argument(
        '-t', '--tasks', type=str, default=None,
        help='Text file with tasks of interest, one task per line. These are the tasks evaluated for subgroup discovery.')
    parser.add_argument(
        '-c', '--config', type=str, default=None,
        help='YAML config file for subgroup discovery parameters and tasks.')
    parser.add_argument(
        '--sample_frac', type=float, default=1.0,
        help='Fraction of the data to use for debugging purposes (0.0, 1.0]')
    return parser.parse_args()

def read_lines(file) -> list:
    with open(file) as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def get_clean_binary_target(outcomes_df, target_outcome, pos_label=0):
    """
    Filter outcomes data and define positive and negative outcomes.

    Returns:
        Tuple with the following:
            - filtered outcomes df
            - clean labels
            - boolean slice used to filter data
    """
    df = outcomes_df.copy()
    all_outcomes = df.columns
    nontarget_outcomes = np.setdiff1d(all_outcomes, target_outcome)
    cases = [
        # Case 1: All outcomes are negative
        np.all([df[col] != pos_label for col in all_outcomes], axis=0),
        # Case 2: Target outcome is positive (regardless of other outcome label)
        (df[target_outcome] == pos_label),
        # Case 3: Target outcome is negative but other outcomes are positive
        np.logical_and(
            (df[target_outcome] != pos_label),
            np.any([df[col] == pos_label for col in nontarget_outcomes], axis=0))
        ]
    labels = [
        abs(1 - pos_label), # No outcomes (negative class)
        pos_label, # Target outcome present (positive class)
        np.nan # Drop negatives with other outcomes
    ]
    outcome_labels = np.select(cases, labels, default=np.nan)
    include = ~np.isnan(outcome_labels)
    return df.loc[include, :], outcome_labels[include], include

def min_max_scale(scores):
    """Scale activations to [0,1] range using min-max scaling"""
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val == min_val:  # Handle edge case
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)

def scale_and_reverse_scores(scores):
    return 1 - min_max_scale(scores)

def safe_roc_auc_score(y_true, y_score, *args, **kwargs):
    """Defaults to 0.5 when there is only one class"""
    unique_labels = set(y_true)
    if len(unique_labels) <= 1:
        # Designed for the edge case "ValueError: Only one class present in y_true. ROC AUC score is not defined in that case."
        # Or the empty vector case (such as for a candidate subgroup that doesn't contain any individuals)
        return 0.5
    else:
        return roc_auc_score(y_true, y_score, *args, **kwargs)

def safe_average_precision_score(y_true, y_score, *args, **kwargs):
    """Defaults to 0.0 when there is only one class or empty.

    For single-class scenarios, precision-recall is undefined.
    A value of 0.0 indicates that no meaningful precision-recall 
    curve exists for this data.
    """
    unique_labels = set(y_true)
    if len(unique_labels) <= 1:
        # Handles case where only one class label is present, in this case the average_precision_score is not informative.
        # Also the case where y_true is empty
        return 0.0
    else:
        return average_precision_score(y_true, y_score, *args, **kwargs)

def safe_auprc(y_true, y_score, *args, **kwargs):
    """Edge case handling for area under the precision-recall curve."""
    unique_labels = set(y_true)
    if len(unique_labels) <= 1:
        return 0.0
    else:
        precision, recall, thresholds = precision_recall_curve(y_true, y_score, *args, **kwargs)
        return auc(recall, precision)

###################################################################################
## Read in of preds, true values, and data used for subgroup disc (metadata.csv) ##
###################################################################################
default_config = {
    'quantile_cuts': [2, 3, 5], # Per previous experiments
    'depth': 4,
    'top_k_percent_data': 20,
    'outcomes': ['bpd_any', 'rop_any', 'ivh_any', 'nec_any'],
    'evaluation_order': ['AUROC', 'AVG Precision'],
    'evaluation_parameters': {
        'AUROC': {
            'alphas': {"bpd_any":0.0575, "rop_any":0.073, "ivh_any":0.0585, "nec_any":0.085},
            'sizes': {"bpd_any":200, "rop_any":300, "ivh_any":100, "nec_any":100}
        },
        'AVG Precision': {
            'alphas': {"bpd_any":0.059, "rop_any":0.06, "ivh_any":0.06, "nec_any":0.025},
            'sizes': {"bpd_any":100, "rop_any":300, "ivh_any":100, "nec_any":100}
        }
    }
}

def main(args):
    results_dir = args.input_directory
    output_dir = args.output_directory
    tasks = args.tasks
    config_file = args.config
    sample_frac = args.sample_frac

    # Process either tasks or a config file that includes tasks
    if tasks is not None:
        outcomes_to_evaluate = read_lines(tasks)
        config = default_config
    elif config_file is not None:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        outcomes_to_evaluate = config['evaluation_outcomes']
        all_outcomes = config['all_outcomes']
    else:
        raise ValueError('Must provide one of the following options: --tasks OR --config')

    # read in previous outputs from the bottleneck layer
    preds = pd.read_csv(results_dir + "bottleneck.csv")
    val_preds = pd.read_csv(results_dir + "valid_bottleneck.csv")
    true_vals = pd.read_csv(results_dir + "true_vals.csv")

    #read in raw data to get actual response for validation data, not currently included in prediction .csv's
    external_true_vals = pd.read_csv("./data/processed/neonatal_conditions.csv", low_memory=False).set_index('row_id')

    #rename predictions columns to be consistent
    val_preds = val_preds.rename(columns={"Unnamed: 0":"row_id"})
    preds = preds.rename(columns={"Unnamed: 0":"row_id"})

    #maintain predictions across individual model runs to calculate Mean + SD
    many_preds = preds.copy().pivot_table(index="row_id",columns="iter",values="bottleneck_unit_0")
    many_val_preds = val_preds.copy().pivot_table(index="row_id",columns="iter",values="bottleneck_unit_0")

    #average over all iteration runs, previously was only taking one
    # groupby and mean() by default should set the "row_id" as the index
    preds = preds.groupby(["row_id"])["bottleneck_unit_0"].mean() #smart way (it is, double checked against stupid for loop method)
    val_preds = val_preds.groupby(["row_id"])["bottleneck_unit_0"].mean() #smart way (it is, double checked against stupid for loop method)
    true_vals = true_vals.groupby(["row_id"])[outcomes_to_evaluate].mean() #smart way (it is, double checked against stupid for loop method)

    #collapse all outcomes to patients x outcomes dataframe
    external_true_vals = external_true_vals[all_outcomes]
    metabolite_labels = pd.read_csv("./config/metabolite_labels.csv")
    data = pd.read_csv("./data/processed/metadata.csv", low_memory=False)

    #######################################################
    ## reduce *_data and *_values to only predicted data ##
    #######################################################

    #subset by observations
    subset_data = data.set_index("row_id").loc[preds.index]
    subset_data = subset_data.drop("gdspid", axis=1)

    # Augment the true vals with the external one
    if set(true_vals.columns) != set(all_outcomes):
        print('Augmenting true values with the whole set of true values.')
        outcomes_available_externally = set(all_outcomes).intersection(external_true_vals.columns)
        augment_outcomes = list(outcomes_available_externally.difference(true_vals.columns))
        true_vals = pd.merge(
            true_vals, external_true_vals[augment_outcomes],
            left_index=True, right_index=True, how='left')
        print(f'Added the following outcomes externally: {augment_outcomes}')

    #subset by observations
    subset_val_data = data.set_index("row_id").loc[val_preds.index]
    subset_val_data = subset_val_data.drop("gdspid", axis=1)

    #subsetting  holdout predictions
    validation_true_vals = external_true_vals.loc[subset_val_data.index, true_vals.columns]

    # OPTIONAL fraction sampling for debugging:
    if sample_frac < 1.0:
        np.random.seed(1234)
        # Sample indices from preds to use throughout
        sample_indices = np.random.choice(
            preds.index,
            size=int(len(preds) * sample_frac),
            replace=False
        )
        # Sample the training data
        preds = preds.loc[sample_indices]
        subset_data = subset_data.loc[sample_indices]
        true_vals = true_vals.loc[sample_indices]
        many_preds = many_preds.loc[sample_indices]

        # Sample the validation data separately
        val_sample_indices = np.random.choice(
            val_preds.index,
            size=int(len(val_preds) * sample_frac),
            replace=False
        )
        val_preds = val_preds.loc[val_sample_indices]
        subset_val_data = subset_val_data.loc[val_sample_indices]
        validation_true_vals = validation_true_vals.loc[val_sample_indices]
        many_val_preds = many_val_preds.loc[val_sample_indices]

        print(f"Sampled training data to {len(preds)} rows")
        print(f"Sampled validation data to {len(val_preds)} rows")

    ###############################
    ## Align all data structures ##
    ###############################
    #check that all indices are the same
    assert (preds.index == subset_data.index).all()
    assert (preds.index == true_vals.index).all()
    assert (preds.index == many_preds.index).all()

    assert (val_preds.index == subset_val_data.index).all()
    assert (val_preds.index == validation_true_vals.index).all()
    assert (val_preds.index == many_val_preds.index).all()

    ########################################################################
    ## manipulate predicted probs into class labels via different cutoffs ##
    ########################################################################
    # NOTE: Since we are interested in identifying HEALTHY individuals
    # The target for prediction will be switched to healthy obs
    true_vals = 1 - true_vals # this along with a change in the 'keep' vector is the only change
    validation_true_vals = 1 - validation_true_vals

    outcome_order = outcomes_to_evaluate
    evaluation_order = config['evaluation_parameters'].keys()
    evaluation_functions = {"AVG Precision": safe_average_precision_score, "AUROC":safe_roc_auc_score}


    # Create a list of dataframes for the top K predictions from each subgroup discovery setting
    top_k_subgroup_predictions = []
    # This second list stores predictions from individual iterations
    top_k_subgroup_preds_iters = []

    for metric in evaluation_order:
        print("starting analysis using - " + metric)
        #
        subgroup_alphas = config['evaluation_parameters'][metric]['alphas']
        subgroup_sizes = config['evaluation_parameters'][metric]['sizes']
        evaluation_metric = evaluation_functions[metric]

        #
        all_results = {}
        iter_results = {}
        #
        #start of loop to collect all data
        for outcome in outcome_order:
            print("starting analysis of - " + outcome)
            #this will create the target matrix, possible targets include true classification, true positivies, ect (based on a CUTOFF value)
            targ = outcome
            pred_type = metric

            # NOTE: limiting to TRUE healthy controls (removing controls with positive co-outcomes)
            # This means our evaluation labels need to be of the following:
            # * CASE: individual has outcome (true_vals[targ] == 0, since the healthy outcome == 1), and this CAN include having other outcome labels present
            # * CASE: Individual does not have outcome (true_vals[targ] == 1 AND does not have any other outcomes)
            #     Here, the original logic checks that all other outcomes have a label of 1
            #     which includes samples where individual DOES have BPD (true_vals[targ] == 0) but does NOT have any other outcomes.
            #     as well as samples where the individual DOES NOT have BPD (true_vals[targ] == 1) and does NOT have any other outcomes.
            # TODO: Isn't this more straightforwardly checked as (true_vals[outcomes].sum(axis=1) == num_outcomes) | (true_vals[targ] == 0)?
            # num_other_outcomes = len(outcome_order) - 1
            # keep = (true_vals[np.setdiff1d(outcomes, [targ])].sum(axis=1) == num_other_outcomes) | (true_vals[targ] == 0)
            # keep_val = (validation_true_vals[np.setdiff1d(outcomes, [targ])].sum(axis=1) == num_other_outcomes) | (validation_true_vals[targ] == 0)

            # TODO: **account for the presence of the non-included label from the set of all outcomes**
            # TODO: Replace this with the more straightfoward function that drops cases where outcome label is negative but co-occurrent outcomes are positive.
            # TODO: In this case the provided outcomes_df MUST include the set of all outcomes to be considered for true positive and true negatives to include for the analysis.
            _, _, keep = get_clean_binary_target(outcomes_df=true_vals, target_outcome=targ, pos_label=0)
            _, _, keep_val = get_clean_binary_target(outcomes_df=validation_true_vals, target_outcome=targ, pos_label=0)

            # Report the difference in keep and keep_val versus the full dataset.
            print(f'Original Input Dataset Size: Subgroup Discovery Training: {len(true_vals)}, Subgroup Discovery Holdout: {len(validation_true_vals)}')
            print(f'After Dropping Controls with other Outcome Labels: Subgroup Discovery Training: {keep.sum()}, Subgroup Discovery Holdout: {keep_val.sum()}')
            print(f'Dropped {len(true_vals) - keep.sum()} samples from Subgroup Discovery Training and {len(validation_true_vals) - keep_val.sum()} samples from Subgroup Discovery Holdout Validation')

            # TODO: Then drop the non-evaluation outcome from the true_val label? (Maybe we don't need to do this since we only evaluate the target anyway)
            #k-fold
            outcome_preds = preds.loc[keep]
            outcome_true_vals = true_vals.loc[keep,:]
            outcome_subset_data = subset_data.loc[keep,:]
            #
            #validation
            val_outcome_preds = val_preds.loc[keep_val]
            validation_outcome_true_vals = validation_true_vals.loc[keep_val,:]
            subset_val_data_outcome = subset_val_data.loc[keep_val,:]
            #
            #iter predictions to calculate SD
            many_outcome_preds = many_preds.loc[keep]
            many_val_outcome_preds = many_val_preds.loc[keep_val]
            #
            #double check that all indices are the same
            assert (preds.index == subset_data.index).all()
            assert (preds.index == true_vals.index).all()
            assert (val_preds.index == subset_val_data.index).all()
            assert (val_preds.index == validation_true_vals.index).all()
            #
            ###########################################################################
            ## massage metabolites and demographic data into quantiles for discovery ##
            ###########################################################################
            temp_data = outcome_subset_data[outcome_subset_data.columns.values[outcome_subset_data.isna().sum() == 0]].copy()
            #
            #constructing list of demographic and metabolomic features to keep
            # TODO: This should probably be a config value.
            keep_features = temp_data.columns.to_series().apply(lambda z: True if ("rc" in z) or (z in [targ+pred_type]) else False)
            temp_data = temp_data[temp_data.columns[keep_features]]
            #
            #compile list of features which need to be transformed into quantiles
            transform = temp_data.apply(
                lambda z: z.name if (temp_data[z.name].dtype == "float64") or ("rc" in z.name) else None).unique()
            #
            searchspace_data = {}
            #
            #transform all data into various quantiles, default: [2,3,5]
            for c in temp_data.columns:
                if c in transform:
                    for q in config['quantile_cuts']:
                        column = f"{c}_q-{q}"
                        searchspace_data[column] = pd.qcut(temp_data[c], q, duplicates="drop").cat.codes
                else:
                    searchspace_data[c] = temp_data[c]
            #
            #
            searchspace_data = pd.DataFrame(searchspace_data)
            #
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
            transform = temp_val_data.apply(lambda z: z.name if (temp_val_data[z.name].dtype == "float64") or ("rc" in z.name) else None).unique()
            #
            searchspace_val_data = {}
            #transform all data into various quantiles, default: [2,3,5]
            for c in temp_val_data.columns:
                if c in transform:
                    for q in config['quantile_cuts']:
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
            # TODO: Test the directionality to evaluate the bottleneck score
            if safe_roc_auc_score(outcome_true_vals[targ].to_numpy(), outcome_preds.to_numpy()) < 0.5:
                outcome_preds = scale_and_reverse_scores(outcome_preds)
                many_outcome_preds = many_outcome_preds.apply(lambda x: scale_and_reverse_scores(x))
                val_outcome_preds = scale_and_reverse_scores(val_outcome_preds)
                many_val_outcome_preds = many_val_outcome_preds.apply(lambda x: scale_and_reverse_scores(x))

            target = ps.PredictionTarget(outcome_true_vals[targ].to_numpy(), outcome_preds.to_numpy(), evaluation_metric)
            searchspace = ps.create_selectors(searchspace_data[searchspace_data.columns[is_metabolite]])
            task = ps.SubgroupDiscoveryTask(
                searchspace_data,
                target,
                searchspace,
                result_set_size=subgroup_sizes[outcome],
                depth=config['depth'],
                qf=ps.PredictionQFNumeric(a=subgroup_alphas[outcome])
            )
            results= ps.BeamSearch(beam_width=subgroup_sizes[outcome]).execute(task)
 
            # EXAMPLES
            # pickle after removing data, if you had to.
            # (otherwise the pickled file would be quite large)
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
            #
            subgroup_desc = results.to_dataframe()["subgroup"]
            #
            data = []
            total_elem = ""

            # NOTE: bool_vec represents the continuously growing merge of all discovered subgroups
            bool_vec = np.full((len(searchspace_data.index)), False)
            for count, elem in enumerate(subgroup_desc):

                # NOTE: bool_vec_inner is the number of individuals matching the current subgroup.
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

                # Update bool_vec to collect the existing merge
                bool_vec = np.logical_or(bool_vec,  bool_vec_inner)

                #collect same general data for top 1:index(elem) subgroups
                # NOTE: This part exists because it's difficult to add custom stats to the dataframe original output
                AUPRC = safe_auprc(outcome_true_vals[targ][bool_vec], outcome_preds[bool_vec])
                subgroup_AUPRC = safe_auprc(outcome_true_vals[targ][bool_vec_inner], outcome_preds[bool_vec_inner])

                #append all data to array
                #compile metrics of performancpe
                if(len(outcome_true_vals[targ][bool_vec]) == outcome_true_vals[targ][bool_vec].sum()):
                    AUROC = np.nan
                    AUROC_sd = np.nan
                    AUROC_mean = np.nan
                    #
                    # Loops like this collect performance of individual model iterations
                    temp_auprc = []
                    for i in range(len(many_preds.columns)):
                        iter_auprc = safe_auprc(outcome_true_vals[targ][bool_vec], many_outcome_preds.iloc[:,i][bool_vec])
                        temp_auprc.append(iter_auprc)
                    #
                    AUPRC_sd = np.std(temp_auprc, ddof=1) #used to make std consistent with R
                    AUPRC_mean = np.mean(temp_auprc)
                    #
                else:
                    AUROC = safe_roc_auc_score(outcome_true_vals[targ][bool_vec], outcome_preds[bool_vec])
                    #
                    temp_auroc = []
                    temp_auprc = []
                    for i in range(len(many_preds.columns)):
                        iter_auprc = safe_auprc(outcome_true_vals[targ][bool_vec], many_outcome_preds.iloc[:,i][bool_vec])
                        temp_auprc.append(iter_auprc)
                        temp_auroc.append(safe_roc_auc_score(outcome_true_vals[targ][bool_vec], many_outcome_preds.iloc[:,i][bool_vec])) #
                    #
                    AUROC_sd = np.std(temp_auroc, ddof=1)
                    AUROC_mean = np.mean(temp_auroc)
                    AUPRC_sd = np.std(temp_auprc, ddof=1)
                    AUPRC_mean = np.mean(temp_auprc)
                #
                #
                # NOTE: The code below is the original edge case for all one labels, adding np.nan to the scores.
                # TODO: However, the inner loops may still fail, and np.nan may still propagate forward, which
                # we should check if we want that.
                # And the check should be more robust - it should check for the presence of the presence of only one label, regardless of positive or negative.
                if(len(outcome_true_vals[targ][bool_vec_inner]) == outcome_true_vals[targ][bool_vec_inner].sum()):
                    subgroup_AUROC = np.nan
                    subgroup_AUROC_sd = np.nan
                    subgroup_AUROC_mean = np.nan
                    #
                    temp_auprc = []
                    for i in range(len(many_preds.columns)):
                        iter_auprc = safe_auprc(outcome_true_vals[targ][bool_vec_inner], many_outcome_preds.iloc[:,i][bool_vec_inner])
                        temp_auprc.append(iter_auprc)
                    #
                    subgroup_AUPRC_sd = np.std(temp_auprc, ddof=1)
                    subgroup_AUPRC_mean = np.mean(temp_auprc)
                    #
                else:
                    subgroup_AUROC = safe_roc_auc_score(outcome_true_vals[targ][bool_vec_inner], outcome_preds[bool_vec_inner])
                    #
                    ##
                    temp_auroc = []
                    temp_auprc = []
                    for i in range(len(many_preds.columns)):
                        iter_auprc = safe_auprc(outcome_true_vals[targ][bool_vec_inner], many_outcome_preds.iloc[:,i][bool_vec_inner])
                        temp_auprc.append(iter_auprc)
                        temp_auroc.append(safe_roc_auc_score(outcome_true_vals[targ][bool_vec_inner], many_outcome_preds.iloc[:,i][bool_vec_inner]))
                    #
                    subgroup_AUROC_sd = np.std(temp_auroc, ddof=1)
                    subgroup_AUROC_mean = np.mean(temp_auroc)
                    subgroup_AUPRC_sd = np.std(temp_auprc, ddof=1)
                    subgroup_AUPRC_mean = np.mean(temp_auprc)
                #
                total_elem = elem + "-OR-" + total_elem
                data.append([total_elem, elem, bool_vec.sum(), bool_vec.sum()/len(searchspace_data.iloc[:,0]), bool_vec_inner.sum(), count, AUPRC, subgroup_AUPRC, AUROC, subgroup_AUROC])
            #
            subgroup_results_df = pd.DataFrame(data, columns=["total group", "subgroup", "size", "% data", "subgroup size", "num_groups", "AUPRC","subgroup AUPRC", "AUROC", "subgroup AUROC"])
            #
            #compile metrics of performancpe
            kfold_AUROC = safe_roc_auc_score(outcome_true_vals[targ], outcome_preds)
            kfold_AUPRC = safe_auprc(outcome_true_vals[targ], outcome_preds)
            #
            temp_auroc = []
            temp_auprc = []
            for i in range(len(many_preds.columns)):
                iter_auprc = safe_auprc(outcome_true_vals[targ], many_outcome_preds.iloc[:,i])
                temp_auprc.append(iter_auprc)
                temp_auroc.append(safe_roc_auc_score(outcome_true_vals[targ], many_outcome_preds.iloc[:,i]))
            #
            kfold_AUROC_sd = np.std(temp_auroc, ddof=1)
            kfold_AUROC_mean = np.mean(temp_auroc)
            kfold_AUPRC_sd = np.std(temp_auprc, ddof=1)
            kfold_AUPRC_mean = np.mean(temp_auprc)

            ################################################################################
            ## Collect AUROC, AUPRC, and other metrics on held out validation predictions ##
            ################################################################################
            #
            data = []
            total_elem = ""

            # NOTE: bool_vec represents the continuously growing merge of all discovered subgroups
            bool_vec = np.full((len(searchspace_val_data.index)), False)
            for count, elem in enumerate(subgroup_desc):
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

                # Subgroup specific calculations will fail if there are no members in the subgroup, which is theoretically possible.
                # precision_recall_curve will raise ValueError: y_true takes value in {} (empty set), downstream error from trying to evaluate empty subgroup.
                if not bool_vec_inner.any(): # No matching records for the subgroup description
                    print(f"Skipping subgroup with no matching records: {elem}")
                    continue # move to next subgroup description

                AUPRC = safe_auprc(validation_outcome_true_vals[targ][bool_vec], val_outcome_preds[bool_vec])
                subgroup_AUPRC = safe_auprc(validation_outcome_true_vals[targ][bool_vec_inner], val_outcome_preds[bool_vec_inner])

                # TODO: Is there better handling of this condition? YES, should handle whether or not only one set of values exists in the
                if(len(validation_outcome_true_vals[targ][bool_vec]) == validation_outcome_true_vals[targ][bool_vec].sum()):
                    AUROC = np.nan
                    AUROC_sd = np.nan
                    AUROC_mean = np.nan
                    #
                    temp_auprc = []
                    for i in range(len(many_preds.columns)):
                        iter_auprc = safe_auprc(validation_outcome_true_vals[targ][bool_vec], many_val_outcome_preds.iloc[:,i][bool_vec])
                        temp_auprc.append(iter_auprc)
                    #
                    AUPRC_sd = np.std(temp_auprc, ddof=1)
                    AUPRC_mean = np.mean(temp_auprc)
                    #
                else:
                    AUROC = safe_roc_auc_score(validation_outcome_true_vals[targ][bool_vec], val_outcome_preds[bool_vec])
                    #
                    temp_auroc = []
                    temp_auprc = []
                    for i in range(len(many_preds.columns)):
                        iter_auprc = safe_auprc(validation_outcome_true_vals[targ][bool_vec], many_val_outcome_preds.iloc[:,i][bool_vec])
                        temp_auprc.append(iter_auprc)
                        temp_auroc.append(safe_roc_auc_score(validation_outcome_true_vals[targ][bool_vec], many_val_outcome_preds.iloc[:,i][bool_vec]))
                    #
                    AUROC_sd = np.std(temp_auroc, ddof=1)
                    AUROC_mean = np.mean(temp_auroc)
                    AUPRC_sd = np.std(temp_auprc, ddof=1)
                    AUPRC_mean = np.mean(temp_auprc)
                #
                #
                # TODO: NOTE: This checks if _within the subgroup itself_ there's all positive labels, which should also be generalized.
                if(len(validation_outcome_true_vals[targ][bool_vec_inner]) == validation_outcome_true_vals[targ][bool_vec_inner].sum()):
                    subgroup_AUROC = np.nan
                    subgroup_AUROC_sd = np.nan
                    subgroup_AUROC_mean = np.nan
                    #
                    temp_auprc = []
                    for i in range(len(many_preds.columns)):
                        iter_auprc = safe_auprc(validation_outcome_true_vals[targ][bool_vec_inner], many_val_outcome_preds.iloc[:,i][bool_vec_inner])
                        temp_auprc.append(iter_auprc)
                    #
                    subgroup_AUPRC_sd = np.std(temp_auprc, ddof=1)
                    subgroup_AUPRC_mean = np.mean(temp_auprc)
                    #
                else:
                    subgroup_AUROC = safe_roc_auc_score(validation_outcome_true_vals[targ][bool_vec_inner], val_outcome_preds[bool_vec_inner])
                    #
                    #calculate sd metrics
                    temp_auroc = []
                    temp_auprc = []
                    for i in range(len(many_preds.columns)):
                        iter_auprc = safe_auprc(validation_outcome_true_vals[targ][bool_vec_inner], many_val_outcome_preds.iloc[:,i][bool_vec_inner])
                        temp_auprc.append(iter_auprc)
                        temp_auroc.append(safe_roc_auc_score(validation_outcome_true_vals[targ][bool_vec_inner], many_val_outcome_preds.iloc[:,i][bool_vec_inner]))
                    #
                    subgroup_AUROC_sd = np.std(temp_auroc, ddof=1)
                    subgroup_AUROC_mean = np.mean(temp_auroc)
                    subgroup_AUPRC_sd = np.std(temp_auprc, ddof=1)
                    subgroup_AUPRC_mean = np.mean(temp_auprc)
                #
                #
                total_elem = elem + "-OR-" + total_elem

                # Fill the existing list of results
                data.append([total_elem, elem, bool_vec.sum(), bool_vec.sum()/len(searchspace_val_data.iloc[:,0]), bool_vec_inner.sum(), count, AUPRC, subgroup_AUPRC, AUROC, subgroup_AUROC])

            # Create a dataframe from the filled results
            subgroup_val_results_df = pd.DataFrame(data, columns=["total group","subgroup","size", "% data", "subgroup size", "num_groups", "AUPRC", "subgroup AUPRC", "AUROC", "subgroup AUROC"])
            print(f"Created validation results dataframe with {len(subgroup_val_results_df)} rows")

            #
            val_AUROC = safe_roc_auc_score(validation_outcome_true_vals[targ], val_outcome_preds)
            val_AUPRC = safe_auprc(validation_outcome_true_vals[targ], val_outcome_preds)
            #
            temp_auroc = []
            temp_auprc = []
            for i in range(len(many_preds.columns)):
                iter_auprc = safe_auprc(validation_outcome_true_vals[targ], many_val_outcome_preds.iloc[:,i])
                temp_auprc.append(iter_auprc)
                temp_auroc.append(safe_roc_auc_score(validation_outcome_true_vals[targ], many_val_outcome_preds.iloc[:,i]))
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
            # Evaluate the top K percentile of data, usually the top 20% of data created by cumulatively merging top subgroups
            topk_thresh_percent = config['top_k_percent_data']
            select = (subgroup_results_df[r"% data"]* 100)

            # Check for edge case for being unable to reach the topk threshold percent in the data!
            reached_topk_percent = any(select >= topk_thresh_percent)
            if not reached_topk_percent:
                max_percent_reached = max(select)
                max_index = select.index[select == max_percent_reached][0]
                print(
                    f"Warning: Cumulative data percentage never reached top K of {topk_thresh_percent}%. "
                    f"The maximum percentage reached is {max_percent_reached:.2f}% at index {max_index}. "
                    f"Selection will still be made based on this closest percentage."
                )
                select_index = max_index
            else:
                select_index = select.index[select == min(select, key=lambda x:abs(x - topk_thresh_percent))][0]

            # Find the index where we get closest to the top K percentile of data
            bool_vec = np.full((len(searchspace_data.index)), False)

            for count, elem in enumerate(subgroup_desc):
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

                    # Here, the ROC thresholds and PR thresholds are saved separately to be able to re-create the curve
                    ROC_tuple_20 = roc_curve(outcome_true_vals[targ][bool_vec], outcome_preds[bool_vec])
                    kfold_AUROC_20 = auc(ROC_tuple_20[0], ROC_tuple_20[1])
                    PR_tuple_20 = precision_recall_curve(outcome_true_vals[targ][bool_vec], outcome_preds[bool_vec])
                    kfold_AUPRC_20 = auc(PR_tuple_20[1], PR_tuple_20[0])
                    #
                    rand_AUROC_20 = roc_auc_score(outcome_true_vals[targ][bool_vec], rand_pred[bool_vec])
                    rand_precision, rand_recall, rand_thresholds = precision_recall_curve(outcome_true_vals[targ][bool_vec], rand_pred[bool_vec])
                    rand_AUPRC_20 = auc(rand_recall, rand_precision)
                    #
                    temp_auroc = []
                    temp_auprc = []
                    for i in range(len(many_preds.columns)):
                        iter_auprc = safe_auprc(outcome_true_vals[targ][bool_vec], many_outcome_preds.iloc[:,i][bool_vec])
                        temp_auprc.append(iter_auprc)
                        temp_auroc.append(safe_roc_auc_score(outcome_true_vals[targ][bool_vec], many_outcome_preds.iloc[:,i][bool_vec]))
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
                    break

            # Repeat with the top K % of data in the validation dataset
            select =  (subgroup_val_results_df["% data"]* 100)
            # Check for edge case for being unable to reach the topk threshold percent in the data!
            reached_topk_percent = any(select >= topk_thresh_percent)
            if not reached_topk_percent:
                max_percent_reached = max(select)
                max_index = select.index[select == max_percent_reached][0]
                print(
                    f"VALIDATION SET: Warning: Cumulative data percentage never reached top K of {topk_thresh_percent}%. "
                    f"The maximum percentage reached is {max_percent_reached:.2f}% at index {max_index}. "
                    f"Selection will still be made based on this closest percentage."
                )
                select_index = max_index
            else:
                select_index = select.index[select == min(select, key=lambda x:abs(x - topk_thresh_percent))][0]
            bool_vec = np.full((len(searchspace_val_data.index)), False)
            #
            for count, elem in enumerate(subgroup_desc):
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

                    # Again, save the actual thresholds for the ROC curve and PR curve to be able to recreate
                    ROC_tuple_20_val = roc_curve(validation_outcome_true_vals[targ][bool_vec], val_outcome_preds[bool_vec])
                    val_AUROC_20 = auc(ROC_tuple_20_val[0], ROC_tuple_20_val[1])
                    PR_tuple_20_val = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec], val_outcome_preds[bool_vec])
                    val_AUPRC_20 = auc(PR_tuple_20_val[1], PR_tuple_20_val[0])
                    #
                    rand_val_AUROC_20 = roc_auc_score(validation_outcome_true_vals[targ][bool_vec], rand_val_pred[bool_vec])
                    rand_precision, rand_recall, rand_thresholds = precision_recall_curve(validation_outcome_true_vals[targ][bool_vec], rand_val_pred[bool_vec])
                    rand_val_AUPRC_20 = auc(rand_recall, rand_precision)
                    #
                    #
                    temp_auroc = []
                    temp_auprc = []
                    for i in range(len(many_preds.columns)):
                        iter_auprc = safe_auprc(validation_outcome_true_vals[targ][bool_vec], many_val_outcome_preds.iloc[:,i][bool_vec])
                        temp_auprc.append(iter_auprc)
                        temp_auroc.append(safe_roc_auc_score(validation_outcome_true_vals[targ][bool_vec], many_val_outcome_preds.iloc[:,i][bool_vec]))
                    #
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
                    break

            #
            iter_results[targ+pred_type] = [kfold_AUROC_mean, kfold_AUROC_sd, val_AUROC_mean, val_AUROC_sd, kfold_AUROC_20_mean, kfold_AUROC_20_sd, val_AUROC_20_mean, val_AUROC_20_sd,
            kfold_AUPRC_mean, kfold_AUPRC_sd, val_AUPRC_mean, val_AUPRC_sd, kfold_AUPRC_20_mean, kfold_AUPRC_20_sd, val_AUPRC_20_mean, val_AUPRC_20_sd]
            #
            all_results[targ+pred_type] = [subgroup_results_df, subgroup_val_results_df, kfold_AUROC, kfold_AUPRC, val_AUROC, val_AUPRC, PR_tuple_20, PR_tuple_20_val, kfold_AUPRC_20, val_AUPRC_20, ROC_tuple_20, ROC_tuple_20_val, kfold_AUROC_20, val_AUROC_20, rand_AUROC, rand_AUPRC, rand_val_AUROC, rand_val_AUPRC, rand_AUROC_20, rand_AUPRC_20, rand_val_AUROC_20, rand_val_AUPRC_20]
        #
        #
        #save to file
        os.makedirs(output_dir, exist_ok=True)
        with open(output_dir + metric + "_bottleneck_results.pkl", "wb") as f:
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
        output.write_excel(
            file=output_dir + metric + "_bottleneck_results.xlsx",
            pred_type=metric,
            all_results=all_results, iter_results=iter_results,
            outcome_order=outcome_order)
        print("writing results to .csv")

    # Save top K subgroup predictions
    top_k_subgroup_predictions = pd.concat(top_k_subgroup_predictions)
    top_k_subgroup_predictions.to_csv(
        output_dir + 'top_k_subgroup_predictions.csv')
    top_k_subgroup_preds_iters = pd.concat(top_k_subgroup_preds_iters)
    top_k_subgroup_preds_iters.to_csv(
        output_dir + 'top_k_subgroup_preds_over_iters.csv')

    # Write out the config that was used, if provided.
    with open(output_dir + 'config.yml', 'w') as f:
        yaml.dump(config, f)


if __name__ == '__main__':
    args = get_args()
    main(args)