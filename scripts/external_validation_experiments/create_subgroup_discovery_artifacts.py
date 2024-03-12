"""Create subgroup discovery artifacts."""
# general imports
import argparse
import json
import logging
from pathlib import PurePath
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np

import joblib
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import KBinsDiscretizer

import xlsxwriter
import pickle
import pysubgroup as ps
from biobank_project.subgroup_discovery import prediction_target


def get_args():
    parser = argparse.ArgumentParser(
        description='Create subgroup discovery artifacts for external validation',
        conflict_handler='resolve')
    parser.add_argument(
        '-i', '--input_predictions_directory', default='./results/deep_mtl/neonatal_bottleneck_validation/ensemble_bottle_1/',
        help='Input predictions directory, should contain bottleneck.csv and true_vals.csv')
    parser.add_argument(
        '-m', '--metabolite_data', default='./data/processed/neonatal_conditions.csv',
        help='Input metabolite profile data for defining subgroup quantiles.')
    parser.add_argument(
        '-c', '--config', default=None,
        help='JSON config file for subgroup discovery procedure.')
    parser.add_argument(
        '-l', '--log_level', default='INFO', help='logger level')
    parser.add_argument(
        '-o', '--output_directory',
        help='Output directory for subgroup discovery artifacts')
    return parser.parse_args()


def main(args):
    results_dir = args.input_predictions_directory
    metabolites_file = args.metabolite_data
    output_dir = PurePath(args.output_directory)
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

    # read in previous outputs from the bottleneck layer
    preds = pd.read_csv(results_dir + "bottleneck.csv")
    preds = preds.rename(columns={"Unnamed: 0":"row_id"})
    true_vals = pd.read_csv(results_dir + "true_vals.csv").set_index('row_id')

    #read in raw data to get actual response for validation data, not currently included in prediction .csv's
    cal_biobank_data = pd.read_csv(metabolites_file, low_memory=False)

    #maintain predictions across individual model runs to calculate Mean + SD
    preds_over_iters = preds.copy().pivot_table(
        index='row_id',columns='iter',values='bottleneck_unit_0')
    preds = preds.groupby(["row_id"])["bottleneck_unit_0"].mean()
    if 'iter' in true_vals.columns:
        true_vals = true_vals[true_vals.iter == 0].drop(
            columns=['fold', 'iter'])
    true_vals = true_vals.loc[preds.index, :]

    #collapse all outcomes to patients x outcomes dataframe
    external_true_vals = cal_biobank_data[["nec_any","rop_any","bpd_any", "ivh_any"]]
    metabolite_labels = pd.read_csv("./config/metabolite_labels.csv")

    # Read in the California metabolite labels
    cal_metabolites = config.get('metabolite_list')
    if cal_metabolites is None:
        with open('./config/expected_metabolite_order.txt') as f:
            cal_metabolites = [l.strip() for l in f.readlines()]

    # Read in metadata
    metadata = pd.read_csv("./data/processed/metadata.csv", low_memory=False)

    #check that all indices are the same
    assert (preds.index == true_vals.index).all()
    assert (preds.index == preds_over_iters.index).all()

    #order is bpd -> rop -> ivh -> nec
    outcome_order = config.get('outcome_order', ["bpd", "rop", "ivh", "nec"])
    evaluation_order = config.get('evaluation_order', ["AUROC", "AVG_Precision"])

    qf_hyperparams = config.get('qf_hyperparams')
    avg_prec_hyperparams = qf_hyperparams['average_precision']
    hyperparam_defaults = {
        'avg_precision': {
            'alpha': {"bpd":0.059, "rop":0.06, "ivh":0.06, "nec":0.025},
            'size': {"bpd":100, "rop":300, "ivh":100, "nec":100}},
        'auroc': {
            'alpha': {"bpd":0.0575, "rop":0.073, "ivh":0.0585, "nec":0.085},
            'size': {"bpd":200, "rop":300, "ivh":100, "nec":100}}
    }
    subgroup_alphas_avg_prec = avg_prec_hyperparams.get('alpha')
    subgroup_sizes_avg_prec = avg_prec_hyperparams.get('size')

    auroc_hyperparams = qf_hyperparams['auroc']
    subgroup_alphas_auroc = auroc_hyperparams.get('alpha')
    subgroup_sizes_auroc = auroc_hyperparams.get('size')

    subgroup_alphas_list = {"AVG_Precision":subgroup_alphas_avg_prec, "AUROC":subgroup_alphas_auroc}
    subgroup_sizes_list = {"AVG_Precision":subgroup_sizes_avg_prec, "AUROC":subgroup_sizes_auroc}

    for qf, alphas in subgroup_alphas_list.items():
        if alphas is None:
            logger.warn(
                f'Alpha values not found in config for {qf}: using defaults.')
            subgroup_alphas_list[qf] = hyperparam_defaults[qf]['alpha']
    for qf, sizes in subgroup_sizes_list.items():
        if sizes is None:
            logger.warn(
                f'Size values not found in config for {qf}: using defaults.')
            subgroup_sizes_list[qf] = hyperparam_defaults[qf]['size']

    evaluation_lists = {"AVG_Precision":average_precision_score, "AUROC":roc_auc_score}

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
            col_annotation = "_sgdisc"

            # NOTE: limit to True healthy controls (removing controls with positive co-outcomes)
            outcome_labels = ["nec_any","rop_any","bpd_any","ivh_any"]
            other_outcomes = np.setdiff1d(outcome_labels, targ)
            in_true_controls = (true_vals[other_outcomes].sum(axis=1) == 0) & (true_vals[targ] == 0)
            in_analysis_set = (in_true_controls) | (true_vals[targ] == 1)

            # NOTE: Since we are interested in identifying HEALTHY individuals
            # The target for prediction will be switched to healthy obs
            healthy_true_vals = 1 - true_vals # this along with a change in the 'in_analysis_set' vector is the only change

            #k-fold
            outcome_preds = preds.loc[in_analysis_set]
            outcome_true_vals = healthy_true_vals.loc[in_analysis_set,:]
            targ_true_vals = outcome_true_vals[targ]

            #iter predictions to calculate SD
            outcome_preds_over_iters = preds_over_iters.loc[in_analysis_set]

            #double check that all indices are the same
            assert (preds.index == true_vals.index).all()

            metab_data = cal_biobank_data[cal_metabolites]
            metab_data = metab_data.loc[preds.index]
            searchspace_input = metab_data[metab_data.columns.values[metab_data.isna().sum() == 0]].copy()
            searchspace_input = searchspace_input.loc[outcome_preds.index]

            #constructing list of demographic and metabolomic features to keep
            # NOTE: Currently only filters on metabolite columns, but extension
            # to other demographics would require different logic
            # e.g., {columns are in metabolites} | searchspace_input.columns.isin(demographics_features)
            in_analysis_set_features = (searchspace_input.columns.isin(cal_metabolites))

            # TODO: Check the searchspace input, which needs to be common to the metabolites?
            # TODO: Or need to incude the config that was used.
            searchspace_input = searchspace_input[searchspace_input.columns[in_analysis_set_features]]

            #compile list of features which need to be transformed into quantiles
            # NOTE: Categorical features need to be protected from quantile transformation
            to_transform = searchspace_input.columns.isin(cal_metabolites) | searchspace_input.apply(is_numeric_dtype)
            cols_to_transform = searchspace_input.columns[to_transform]

            searchspace_quantiles = [2,3,4]
            transform_input = searchspace_input[cols_to_transform]
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
                r'_q.*$', '').isin(cal_metabolites)

            # INIT subgroup discovery objects and procedure
            target = prediction_target.PredictionTarget(
                targ_true_vals.to_numpy(), outcome_preds.to_numpy(),
                evaluation_metric)
            searchspace = ps.create_selectors(searchspace_data[searchspace_data.columns[is_metabolite]])
            task = ps.SubgroupDiscoveryTask(
                searchspace_data,
                target,
                searchspace,
                result_set_size=subgroup_sizes[outcome],
                depth=3,
                qf=prediction_target.PredictionQFNumeric(a=subgroup_alphas[outcome])
            )

            results = ps.BeamSearch(beam_width=subgroup_sizes[outcome]).execute(task)
            all_subgroups = results.results

            subgroup_desc_df = results.to_dataframe()

            # Then the usage is like:
            # for sg_num, sg in enumerate(all_subgroups):
            #     sg_quality, sg_description, qf = sg
            #     sg_mask, sg_size = ps.get_cover_array_and_size(sg_description, data=searchspace_data)

            # Write out results:
            subgroup_desc_df.to_csv(
                str(output_dir.joinpath(
                    f'{metric}_{outcome}_subgroup_discovery_results.csv')))

            # Write out specific objects
            joblib.dump(
                discretizers, str(output_dir.joinpath(
                    f'{metric}_{outcome}_discretizers.joblib')))
            joblib.dump(
                results, str(output_dir.joinpath(
                    f'{metric}_{outcome}_subgroup_discovery_combined_object.joblib')))
            joblib.dump(
                all_subgroups, str(output_dir.joinpath(
                    f'{metric}_{outcome}_all_subgroups.joblib')))


if __name__ == '__main__':
    args = get_args()
    main(args)
