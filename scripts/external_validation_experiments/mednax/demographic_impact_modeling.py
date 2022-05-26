"""Modeling the impact of demographics on metabolite data."""
import argparse
import json
from pathlib import PurePath

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from biobank_project.io import read_lines


def main():
    # TODO: Read in the metadata from the California Biobank and Mednax
    # Read in the Mednax data and get the same dimensions as the California dataset
    # Follows previous logic for PCA
    mednax = pd.read_csv(
        './external_validation/mednax/processed/mednax_metabolites_cal_names.csv')
    mednax.drop(columns=['ID'], inplace=True)
    mednax.set_index('QuestionsRCode', inplace=True)

    # Prepare data to perform dimensionality reduction using the California subset.
    with open('./config/expected_metabolite_order.txt') as f:
        cal_metabolites = [l.strip() for l in f.readlines()]

    cal_biobank_data = (pd.read_csv(
        "./data/processed/neonatal_conditions.csv", low_memory=False)
        .set_index('row_id'))
    meta = (pd.read_csv('./data/processed/metadata.csv', low_memory=False)
        .set_index('row_id'))

    with open('./config/gestational_age_ranges.txt') as f:
        preterm_ranges = [l.strip() for l in f.readlines()]
    preterm_meta = meta[meta.gacat.isin(preterm_ranges)]
    preterm_metabolites = cal_biobank_data[cal_metabolites].loc[preterm_meta.index]

    # Set up metadata
    cal_meta = preterm_meta.copy()
    med_meta = pd.read_csv(
        './external_validation/mednax/processed/mednax_demo_meta_other.csv')
    med_meta.set_index('QuestionsRCode', inplace=True)

    # Harmonize metabolite features
    # Need to drop the sparse metabolites between the two
    cal_metab = preterm_metabolites.copy()
    cal_metab = cal_metab.loc[cal_meta.index]
    cal_metab['dataset'] = 'California Biobank'
    mednax['dataset'] = 'Mednax'

    # Drop sparse metabolites
    # thresh=len(df) * 0.95 implies that the column must be non-missing in 95% of the df
    mednax.dropna(thresh=len(mednax) * 0.95, axis=1, inplace=True)
    cal_metab.dropna(thresh=len(cal_metab) * 0.95, axis=1, inplace=True)

    common_metab = mednax.columns.intersection(cal_metab.columns)
    combined_metabolites = pd.concat(
        [cal_metab[common_metab], mednax[common_metab]])
    combined_metabolites.dropna(inplace=True)

    # Harmonize potential outcomes as dataset source and race-ethnicity
    # Use maternal race in the California biobank dataset as an indicator for race-ethnicity
    cal_raceethnicity_codes = {
        1: 'White',
        2: 'Black',
        3: 'Asian',
        4: 'Pacific Islander',
        5: 'Hispanic',
        6: 'American Indian',
        7: 'Other',
        99: np.nan}
    cal_meta['race'] = cal_meta.mrace_catm.replace(to_replace=cal_raceethnicity_codes)
    med_meta['race'] = med_meta['Race']
    meta_cols = ['race']
    combined_meta = pd.concat(
        [cal_meta[meta_cols], med_meta[meta_cols]])
    outcomes = ['race', 'dataset']

    # Combine and split the two datasets
    combined = pd.merge(
        combined_metabolites, combined_meta, left_index=True, right_index=True)
    features = combined.columns[~combined.columns.isin(outcomes)]

    # Set up data splitting
    random_state_int = 100
    rs = np.random.RandomState(seed=random_state_int)
    n_folds = 5
    n_kfold_repeats = 10
    kf = StratifiedKFold(
        n_splits=n_folds, random_state=rs,
        shuffle=True)

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
    scorer = make_scorer(
        metrics.f1_score, average='weighted', zero_division=0)
    search_args = {
        'n_iter': 100,
        'cv': 5,
        'scoring': scorer,
        'random_state': random_state_int,
        'n_jobs': n_jobs,
        'verbose': 1
    }
    # TODO: Set up multiclass for race-ethnicity signal
    rf = RandomForestClassifier(
        class_weight='balanced_subsample',
        random_state=rs)
    rfcv = RandomizedSearchCV(
        rf, param_distributions=rf_grid, **search_args)

    # TODO: Implement other models
    en_cv = None
    # xgboostcv = None
    models = {}

    # TODO: Write out K-Fold test set predictions
    # Structures for storing K-Fold results
    kfold_dfs = []
    kfold_val_dfs = []
    feat_import_dfs = []

    # Begin nested K-Fold cross validation procedure
    X_data, outcome_data = combined[features], combined[outcomes]
    Y_labels = outcome_data['race']
    Y_data = pd.get_dummies(Y_labels)
    # TODO: KFold on California, predict race-ethnicity on Mednax
    # X_data = combined[combined.dataset == 'California Biobank'][features]
    # outcome_data = combined[combined.dataset == 'California Biobank']
    # Y_labels = outcome_data['race']
    # Y_data = pd.get_dummies(Y_labels)

    validation_set = False
    X_val, Y_val = (None, None)
    # X_val = combined[combined.dataset == 'Mednax'][features]
    # Y_val_labels = combined[combined.dataset == 'Mednax']['race']
    # Y_val = pd.get_dummies(Y_val_labels)
    # Y_val = Y_val[Y_data.columns]

    # TODO: Experiment with single task models
    for o in Y_data.columns:
        for i in range(n_kfold_repeats):
            splits = kf.split(X_data, outcome_data['dataset'])
            for fold_num, (train_ix, test_ix) in enumerate(splits):
                #X_train, Y_train = X_data.iloc[train_ix, :], Y_data.iloc[train_ix, :]
                #X_test, Y_test = X_data.iloc[test_ix, :], Y_data.iloc[test_ix, :]
                # TODO: Change the Y slicing to be specific to one outcome
                Y_specific_outcome = Y_data[[o]]
                X_train, Y_train = X_data.iloc[train_ix, :], Y_specific_outcome.iloc[train_ix, :].values.ravel()
                X_test, Y_test = X_data.iloc[test_ix, :], Y_specific_outcome.iloc[test_ix, :]

                # Scale and transform
                scaler = StandardScaler()
                rfcv.fit(scaler.fit_transform(X_train.values), Y_train)
                probs = rfcv.predict_proba(
                    scaler.transform(X_test))
                test_preds_array = probs[:, 1]
                # TODO: Commented out code is for multioutput
                #test_preds_array = np.stack(
                #    [probs[:, 1] for probs in test_preds], axis=1)
                fold_preds_df = pd.DataFrame(
                    test_preds_array, index=X_test.index,
                    columns=[f'{o}_prob'])
                    #columns=[f'{c}_prob' for c in Y_data.columns])
                fold_preds_df = pd.merge(
                    fold_preds_df, Y_test, left_index=True, right_index=True)
                fold_preds_df['fold'], fold_preds_df['iter'] = fold_num, i
                kfold_dfs.append(fold_preds_df)

                # Repeat for validation
                if validation_set is True:
                    val_preds = rfcv.predict_proba(
                        scaler.transform(X_val))
                    val_preds_array = np.stack(
                        [probs[:, 1] for probs in val_preds], axis=1)
                    fold_val_preds_df = pd.DataFrame(
                        val_preds_array, index=X_val.index,
                        columns=[f'{c}_prob' for c in Y_val.columns])
                    fold_val_preds_df = pd.merge(
                        fold_val_preds_df, Y_val, left_index=True, right_index=True)
                    fold_val_preds_df['fold'], fold_val_preds_df['iter'] = fold_num, i
                    kfold_val_dfs.append(fold_val_preds_df)

                # TODO: Write out feature importance
                feat_import = rfcv.best_estimator_.feature_importances_
                fi_df = pd.DataFrame.from_dict(
                    {'feature': X_data.columns.tolist(),
                    'gini_importance': feat_import})
                fi_df['fold'], fi_df['iter'] = fold_num, i
                fi_df['outcome'] = o
                feat_import_dfs.append(fi_df)

    # Assemble dataframe of results.
    kfold_results_df = pd.concat(kfold_dfs)
    kfold_results_df.index.name = 'index'
    kfold_results_df.to_csv(
        './results/external_validation/cal_mednax_raceethnicity_singletask.csv')

    if validation_set is True:
        kfold_val_results_df = pd.concat(kfold_val_dfs)
        kfold_val_results_df.index.name = 'index'
        kfold_val_results_df.to_csv(
            './results/external_validation/cal_mednax_raceethnicity_val.csv')

    kfold_feat_import_df = pd.concat(feat_import_dfs)
    kfold_feat_import_df.to_csv(
        './results/external_validation/cal_mednax_raceethnicity_singletask_feature_importance.csv', index=False)

if __name__ == '__main__':
    main()
