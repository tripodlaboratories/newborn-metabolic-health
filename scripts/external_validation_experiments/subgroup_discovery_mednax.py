"""Replicate subgroup discovery algorithm in Mednax"""
import argparse
import json
from pathlib import PurePath

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from biobank_project.io import read_lines


def main():
    # Read in subgroup discovery results from the Cal Biobank Dataset
    sd_dir = PurePath('./results/subgroup_discovery/metabolic_health_index/')
    top_k_preds = pd.read_csv(
        sd_dir.joinpath('top_k_subgroup_predictions.csv'))
    top_k_preds.set_index('row_id', inplace=True)

    # Match with metabolites from the California State Biobank
    cal_biobank_dir = PurePath('./data/processed/')
    cal_biobank_metab = pd.read_csv(
        cal_biobank_dir.joinpath('neonatal_conditions.csv'))

    # Read in Mednax metabolites
    mednax_dir = PurePath('./external_validation/mednax/')
    mednax_metab = pd.read_csv(
        mednax_dir.joinpath('processed', 'mednax_metabolites.csv'))
    mednax_metab = (mednax_metab
        .drop(columns='ID')
        .set_index('QuestionsRCode'))

    # Note: Metabolites need to match following the matching logic used for
    # model training
    match_df = pd.read_csv(
        mednax_dir.joinpath(
            'metabolite_matching',
            'calbiobank_mednax_metab_matching.csv'))
    metab_intersect = match_df[
        ~pd.isna(match_df.mednax) | ~pd.isna(match_df.derived_ratio_name)]
    cal_biobank_metab_final = read_lines(
        './config/expected_metabolite_order.txt')
    metab_intersect = metab_intersect[
        metab_intersect.cal_biobank.isin(cal_biobank_metab_final)]
    cal_biobank_metab = cal_biobank_metab[metab_intersect.cal_biobank]
    metab_intersect['mednax_match'] = metab_intersect.mednax.fillna(metab_intersect.derived_ratio_name)
    mednax_metab = mednax_metab[metab_intersect.mednax_match].copy()
    mednax_metab.dropna(inplace=True)

    samples_with_inf = np.isinf(mednax_metab).any(axis=1)
    n_samples_with_inf = samples_with_inf.sum()
    mednax_metab = mednax_metab[~samples_with_inf]

    # Recode as classification problem for top 20% of subgroups or not
    meta = pd.read_csv(
        cal_biobank_dir.joinpath('metadata.csv'), low_memory=False)
    meta.set_index('row_id', inplace=True)
    ga_ranges = read_lines('./config/gestational_age_ranges.txt')
    preterm_ids = meta[meta.gacat.isin(ga_ranges)].index
    cal_biobank_metab = cal_biobank_metab[
        cal_biobank_metab.index.isin(preterm_ids)]
    cal_biobank_metab.dropna(inplace=True)
    isin_top_subgroup = cal_biobank_metab.index.isin(
        top_k_preds.index).astype('int')

    # K-Fold CV for Random Forest predictions
    random_state_int = 101
    rs = np.random.RandomState(random_state_int)
    n_folds = 5
    n_kfold_repeats = 10
    kf = KFold(n_splits=n_folds, random_state=rs, shuffle=True)

    # Set up randomized search for RF hyperparameters
    # Hyperparameter tuning for random forest
    n_estimators = [50, 100, 200, 500, 1000]
    # Maximum number of levels in tree
    max_depth = [10, 30, 50, 70, 90, 100]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 10]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    rf_grid = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap}

    # Multiprocessing and other iterations of randomized search
    n_jobs = 30
    random_search_args = {
        'n_iter': 100,
        'cv': 5,
        'scoring': 'balanced_accuracy',
        'random_state': random_state_int,
        'n_jobs': n_jobs,
        'verbose': 1
    }
    rf = RandomForestClassifier()
    rfcv = RandomizedSearchCV(
        rf, param_distributions=rf_grid, **random_search_args)

    data_X = cal_biobank_metab
    data_y = isin_top_subgroup

    # Structures for storing K-Fold results
    kfold_dfs = []
    kfold_params = []

    # Begin nested K-Fold cross validation procedure
    for i in range(n_kfold_repeats):
        splits = kf.split(data_X, data_y)
        for fold_num, (train_ix, test_ix) in enumerate(kf.split(data_X, data_y)):
            X_train, y_train = data_X.iloc[train_ix, :], data_y[train_ix]
            X_test, y_test = data_X.iloc[test_ix, :], data_y[test_ix]

            # Scale and transform
            scaler = StandardScaler()
            rfcv.fit(scaler.fit_transform(X_train.values), y_train)

            # Store the hyperparameters from the best runs
            params = rfcv.best_params_
            params['fold'] = fold_num
            params['iter'] = i
            kfold_params.append(params)

            # Make predictions on the test set to understand how good we are
            # at predicting the top 20% of subgroups
            sg_test_preds = rfcv.predict_proba(scaler.transform(X_test))
            fold_preds_df = pd.DataFrame(
                sg_test_preds, index=X_test.index,
                columns=['outoftopk_prob', 'topk_prob'])
            fold_preds_df['isin_top_subgroup'] = y_test
            fold_preds_df['fold'], fold_preds_df['iter'] = fold_num, i
            kfold_dfs.append(fold_preds_df)

    # Assemble dataframe of results.
    kfold_results_df = pd.concat(kfold_dfs)
    kfold_params_df = pd.DataFrame(kfold_params)

    # Use the consensus best hyperparameters across K-Fold to then test
    # on the Mednax dataset
    sd_output_dir = PurePath('./results/external_validation/mednax/')
    kfold_results_df.to_csv(
        sd_output_dir.joinpath('calbiobank_subgroup_kfold_preds.csv'))
    kfold_params_df.to_csv(
        sd_output_dir.joinpath('calbiobank_subgroup_kfold_params.csv'))

    # Determine the best parameters across the K-Fold procedure by counts
    # in each fold and across
    kfold_param_counts = (kfold_params_df
        .drop(columns='fold')
        .groupby(['iter'])
        .agg(lambda x: x.value_counts().idxmax()))
    final_params = kfold_param_counts.apply(
        lambda x: x.value_counts().idxmax()).to_dict()

    # Then set these as parameters in the new model, training across
    # a holdout validation?
    new_model = RandomForestClassifier(**final_params)
    new_model.fit(StandardScaler().fit_transform(data_X.values), data_y)

    # Once model is fit apply the model to the mednax data
    mednax_sg_preds = new_model.predict_proba(
        StandardScaler().fit_transform(mednax_metab.values))
    mednax_sg_preds_df = pd.DataFrame(
        mednax_sg_preds, index=mednax_metab.index,
        columns=['outoftopk_prob', 'topk_prob'])

    # Write out the subgroup top 20 predictions to file
    mednax_sg_preds_df.to_csv(
        sd_output_dir.joinpath('mednax_subgroup_preds.csv'))


if __name__ == '__main__':
    main()

