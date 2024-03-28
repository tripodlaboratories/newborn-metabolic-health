"""Score metabolites based on quantile subgroups."""
import argparse
from functools import reduce
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, auc
from sklearn.preprocessing import KBinsDiscretizer
import yaml


DEFAULT_SCORE_FUNCTIONS={'AUROC': roc_auc_score,
                         'AUPRC': average_precision_score}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input',
        help='Input dataframe of metabolite values + outcomes')
    parser.add_argument(
        '-c', '--column_spec',
        help='YAML denoting which columns are [metabolites], [outcomes], or [ids]')
    parser.add_argument(
        '--preds',
        help='Dataframe with model predictions')
    parser.add_argument(
        '--preds_name', default='bottleneck_unit_0',
        help='Column name of the health index model predictions to use')
    parser.add_argument(
        '-o', '--output_dir', help='Output directory for results.')
    return parser.parse_args()


def prc_auc_score():
    # TODO: Implement prcurve area with same API as average_precision_score
    """Maybe it's better to get PR Curve by trapezoidal rule for direct comparison."""
    pass


def score_metab_subgroups(df,
                          metab_cols,
                          outcomes,
                          score_functions):
    scores = []
    df_qbins = derive_quantile_bins(df, metabs_to_bin=metab_cols)
    df_qbins_outcomes = pd.merge(
        df_qbins, df.drop(metab_cols, axis=1),
        left_index=True, right_index=True)

    for metab_quantile in df_qbins.columns:
        for outcome in outcomes:
            specific_fn_scores = []
            for sf_label, sf in score_functions.items():
                # Use metabolite quantile-defined group to evaluate model
                # performance
                score = df_qbins_outcomes.groupby(
                    [metab_quantile]).apply(
                    score_outcome, score_function=sf, outcome=outcome)
                score.name = f'{sf_label}'
                score_df = pd.DataFrame(score)
                score_df.reset_index(inplace=True)
                score_df['outcome'] = outcome

                # Drop NA values from the score_df, which should be
                # resulting from N=1 subgroups or other errors in scoring.
                score_df.dropna(inplace=True)
                specific_fn_scores.append(score_df)
            scores.append(reduce(
                lambda df1, df2: pd.merge(df1, df2, on=[metab_quantile, 'outcome']),
                specific_fn_scores))
    return scores, df_qbins_outcomes


def derive_quantile_bins(df, metabs_to_bin, n_bins=[3, 5]):
    bin_results = []
    for b in n_bins:
        binner = KBinsDiscretizer(n_bins=b, encode='ordinal')
        binned_vals = pd.DataFrame(
            binner.fit_transform(df[metabs_to_bin]),
            columns=[f'{m}_quant_{b}' for m in metabs_to_bin],
            index=df.index)
        bin_results.append(binned_vals)
    return reduce(lambda df1, df2: pd.merge(df1, df2, left_index=True, right_index=True), bin_results)


def score_outcome(df, score_function, outcome='health_indicator'):
    # Most score functions are not defined or meaningful for subgroup N=1
    if df.shape[0] == 1:
        score = np.nan
    else:
        try:
            # Health index can be used to score against 0/1 healthy outcome against
            # all outcomes
            if outcome == 'health_indicator':
                score = score_function(df['health_indicator'], df['health_index'])
            # Otherwise the health index can be used to score against the absence of
            # a specific outcome, where absence of outcome is the 1 label.
            else:
                scoring_df = df[[outcome, 'num_outcomes', 'health_index']]
                scoring_df = scoring_df[~((scoring_df[outcome] == 0) & (scoring_df['num_outcomes'] > 0))]
                scoring_df['no_outcome'] = scoring_df[outcome].replace({0: 1, 1: 0})
                score = score_function(scoring_df['no_outcome'], scoring_df['health_index'])
        except ValueError:
            # Other exceptions in calculating the score function
            score = np.nan
    return score


def drop_sparse_cols(df, sparsity_threshold=0.25):
    col_sparsity = df.apply(np.isnan).sum() / len(df)
    sparse_cols = col_sparsity > sparsity_threshold
    return df.loc[:, ~sparse_cols].copy()


def drop_na_and_check(df):
    original_n = len(df)
    without_na = df.dropna()
    dropped_n = len(without_na)
    percent_dropped = (original_n - dropped_n) / original_n

    if percent_dropped > 0.10:
        warnings.warn(f'Percent dropped from metabolite data: {percent_dropped:.2f}')

    return without_na

def draw_heatmap(*args, **kwargs):
    pass


def score_quantile_performance(df,
                               quantile_cols,
                               score_fns: dict={'AUROC': roc_auc_score, 'AUPRC': average_precision_score},
                               outcome='health_indicator'):
    score_results = []
    for score_name, score_fun in score_fns.items():
        scored_result = df.groupby(quantile_cols).apply(
            score_outcome, score_function=score_fun, outcome=outcome)
        scored_result.name = score_name
        scored_result.reset_index(inplace=True)
        score_results.append(scored_result)

    all_scores = reduce(
        lambda df1, df2: pd.merge(df1, df2, on=quantile_cols), score_results)
    return all_scores


def encode_health_indicator(num_outcomes):
    if num_outcomes == 0:
        # Where 1 corresponds to the absence of any labeled outcomes
        return 1
    else:
        return 0


def add_health_indicator_col(df, outcome_cols):
    condition_counts = df[outcome_cols].copy()
    condition_counts['num_outcomes'] = condition_counts.sum(axis=1)
    condition_counts['health_indicator'] = condition_counts.num_outcomes.apply(
        encode_health_indicator)
    return pd.merge(df.drop(outcome_cols, axis=1), condition_counts, left_index=True, right_index=True)


def average_index_predictions(preds, preds_name):
    # Predictions need to be averaged across folds and iterations.
    # In this groupby(), we convert the Series to DataFrame because downstream
    # DataFrame methods are used on it (e.g., rename())
    return preds[preds_name].groupby(preds.index).mean().to_frame()


def main(args):
    # Process args
    metab_file = args.input
    preds_file = args.preds
    preds_name = args.preds_name
    col_spec_file = args.column_spec
    out_dir = args.output_dir

    with open(col_spec_file, 'rb') as f:
        col_spec = yaml.safe_load(f)

    # Read in metabolite outcomes and healh index model predictions
    health_index_preds = pd.read_csv(preds_file)
    metab_outcomes = pd.read_csv(metab_file)

    # Set index value for later merging, pullling from pre-specified index name
    id_col = col_spec.get('id', None)
    try:
        health_index_preds.set_index(id_col, inplace=True)
    except KeyError:
        health_index_preds.rename(columns={'Unnamed: 0': id_col}, inplace=True)
        health_index_preds.set_index(id_col, inplace=True)
    health_index_preds = average_index_predictions(health_index_preds, preds_name)
    metab_outcomes.set_index(id_col, inplace=True)

    metabs = col_spec.get('metabolites', None)
    outcomes = col_spec.get('outcomes', None)
    metab_outcomes = drop_sparse_cols(metab_outcomes)
    metabs = list(set(metabs).intersection(metab_outcomes.columns))
    metab_outcomes = drop_na_and_check(metab_outcomes)
    metab_condition_counts = add_health_indicator_col(
        metab_outcomes, outcome_cols=outcomes)
    metab_condition_counts = pd.merge(
        metab_condition_counts,
        health_index_preds.rename(columns={preds_name: 'health_index'}),
        left_index=True, right_index=True)
    score_dfs, qbins = score_metab_subgroups(
        metab_condition_counts, metab_cols=metabs, outcomes=outcomes,
        score_functions=DEFAULT_SCORE_FUNCTIONS)
    metab_scores_df = pd.concat(score_dfs, axis=0)
    metab_scores_long = pd.melt(
        metab_scores_df, id_vars=['outcome', 'AUROC', 'AUPRC'],
        var_name='quantile_split', value_name='quantile_assignment')
    metab_scores_long['metab'] = metab_scores_long.quantile_split.str.replace(
        '_quant_[0-9]{1}', '', regex=True)

    # Output CSVs
    metab_scores_long.to_csv(f'{out_dir}metab_scores.csv', index=False)
    qbins.to_csv(f'{out_dir}quantile_assignments.csv')


if __name__ == '__main__':
    args = get_args()
    main(args)

