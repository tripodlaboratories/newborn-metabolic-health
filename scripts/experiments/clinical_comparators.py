# TODO: Compare against regression models using clinical comparators
import argparse
import logging
import os
from pathlib import PurePath
import re

import numpy as np
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from torch import optim, Tensor
from torch.nn import BCEWithLogitsLoss, MSELoss

from biobank_project.deep_mtl.training import utils
from biobank_project.sklearn_interface import kfold, handlers
from biobank_project.deep_mtl.sampling import MajorityDownsampler


def get_argparser():
    parser = argparse.ArgumentParser(
        description='Script for running deep multi-task experiments.',
        conflict_handler='resolve')
    parser.add_argument(
        '-i', '--input', type=str, help='input file with features and outcomes',
        required=True, metavar='INPUT_FILE', dest='input_file')
    parser.add_argument(
        '-o', '--output', type=str, help='output directory for results',
        required=True, metavar='OUTPUT_DIR', dest='output_dir')
    parser.add_argument(
        '-t', '--tasks', type=str,
        help='Text file with tasks of interest, one task per line.')
    parser.add_argument(
        '-n', '--n_iter', type=int, help='number of cross validation iterations to perform',
        default=10)
    parser.add_argument(
        '--cases_only', action='store_true',
        help='perform model training on cases only, ignoring controls.')
    parser.add_argument(
        '--single_condition', action='store_true',
        help='perform model training on cases with only one condition')
    parser.add_argument(
        '-v', '--validate', action='store_true',
        help='split data in train/test/validate.')
    parser.add_argument(
        '--n_jobs', type=int,
        help="--n_jobs argument for sklearn models.")
    return parser


def read_lines(file) -> list:
    with open(file) as f:
        lines = f.readlines()
    return [l.strip() for l in lines]


def write_results(results: dict, model_output_dir: PurePath):
    for result_name, results_df in results.items():
        if result_name == 'preds':
            filename = model_output_dir.joinpath(result_name + '.csv.gz')
            results_df.to_csv(filename, compression='gzip')
        else:
            filename = model_output_dir.joinpath(result_name + '.csv')
            results_df.to_csv(filename)


def main(args):
    logging.basicConfig(
        level=os.environ.get('LOGLEVEL', 'INFO'),
        format='%(asctime)s.%(msecs)03d %(name)s %(levelname)s:%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
        )
    logger = logging.getLogger('model_comparison_script')

    input_file = args.input_file
    output_dir = PurePath(args.output_dir)
    tasks = args.tasks
    n_iter = args.n_iter
    cases_only = args.cases_only
    single_condition = args.single_condition
    validate = args.validate
    n_jobs = args.n_jobs

    # Read in data
    input_data = pd.read_csv(input_file, low_memory=False)
    input_data.set_index('row_id', inplace=True)
    metadata = pd.read_csv('./data/processed/metadata.csv', low_memory=False)
    metadata.set_index('row_id', inplace=True)
    outcomes = read_lines(tasks)

    # Subset data based on gestational ages used
    included_ga_range = read_lines('./config/gestational_age_ranges.txt')
    included_metadata = metadata[metadata['gacat'].isin(included_ga_range)]
    input_data = input_data.loc[included_metadata.index, :]
    assert sorted(input_data.index) == sorted(included_metadata.index)

    # Merge in input data and metadata
    input_data = pd.merge(
        input_data, included_metadata,
        left_index=True, right_index=True, how='inner')

    # Recode infant sex encoding to {0,1} (in original data it is {1,2}
    # 0: males, 1: females
    input_data['infant_sex'] = input_data['sex3'] - 1

    # Organize variables into feature sets for specific types of model input
    feature_sets = {
        'minimal_vars': ['gacat', 'bwtcat'],
        'additional_risk_vars': ['gacat', 'bwtcat', 'infant_sex', 'ap1cat', 'ap5cat']
    }
    # Collect all feature names for subsetting data
    all_features = [f for features in feature_sets.values() for f in features]
    all_features = list(set(all_features))

    # Drop sparse columns
    input_data.dropna(thresh=len(input_data) / 2, axis=1, inplace=True)

    # Handling for cases only
    if cases_only is True:
        input_data['total_conditions'] = input_data[outcomes].sum(axis=1)
        data_subset = input_data[input_data['total_conditions'] >= 1].drop(
            columns=['total_conditions'])
        if data_subset.shape[0] >= input_data.shape[0]:
            logger.warn('--cases_only flag did not reduce rows of data.')
        input_data = data_subset
        assert 'total_conditions' not in input_data

    if single_condition is True:
        input_data['total_conditions'] = input_data[outcomes].sum(axis=1)
        data_subset = input_data[input_data['total_conditions'] == 1].drop(
            column=['total_conditions'])
        if data_subset.shape[0] >= input_data.shape[0]:
            logger.warn('--single_condition flag did not reduce rows of data.')
        input_data = data_subset
        assert 'total_conditions' not in input_data

    # Convert categorical labels of birthweight and gestational age to numeric
    input_data['gacat'] = input_data['gacat'].apply(
        lambda x: float(re.sub(r'^[0-9]*_', '', x)) - 0.5)
    assert pd.api.types.is_float_dtype(input_data['gacat'])
    assert input_data['gacat'].min() > 19

    input_data['bwtcat'] = input_data['bwtcat'].apply(
        lambda x: float(re.sub(r'^[0-9]*_', '', x)) - 24)
    assert pd.api.types.is_float_dtype(input_data['bwtcat'])
    assert input_data['bwtcat'].min() > 400

    # Recode Apgar variables to remove 99's as missing
    input_data.replace(
        {'ap1cat': {99: np.nan},
         'ap5cat': {99: np.nan}
         }, inplace=True)

    # Drop NA rows (and check the impact on dataframe size).
    modeling_data = input_data[all_features + outcomes].copy()
    modeling_data.dropna(inplace=True)

    # Split data into X and Y
    data_X = modeling_data[all_features]
    data_Y = modeling_data[outcomes]

    if validate is True:
        logger.info('Setting up experiment to use holdout validation.')
        data_X, X_valid, data_Y, Y_valid = train_test_split(data_X, data_Y, random_state=101)

    # Set up single task classification and regression models from sklearn
    # choices.
    utils.seed_torch(101)

    # TODO: Setup model training for widespread model comparison
    # TODO: Implement RandomForest, XGBoost, CatBoost?
    # Model specific hyperparameters for logistic regression models
    l1_ratio_range = [0.1, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    num_C_vals = 20
    ridge_alphas = [0.1, 0.5, 1.0, 10.0]
    max_iter = 10000
    n_cv_folds = 5
    classification_scoring_metric='roc_auc'

    # Model specific hyperparameters for ensemble methods
    models = {
        'en': LogisticRegressionCV(
            penalty='elasticnet', cv=n_cv_folds,
            scoring=classification_scoring_metric,
            l1_ratios=l1_ratio_range, solver='saga',
            Cs=num_C_vals, n_jobs=n_jobs, max_iter=max_iter),
        'lasso': LogisticRegressionCV(
            penalty='l1', solver='saga', cv=n_cv_folds,
            scoring=classification_scoring_metric,
            Cs=num_C_vals, n_jobs=n_jobs, max_iter=max_iter),
        'lr': LogisticRegression(
            penalty='none', max_iter=max_iter)
        }

    # TODO: Set up model training
    n_folds = 10
    resampler = MajorityDownsampler(random_state=101)

    # TODO: Experimental: Which models are inherently multitask, and which need
    # the help of the MultiOutputClassifier?
    train_args = {'colnames': data_Y.columns}

    # TODO: Iterate over different feature sets and train models
    for feat_set_name, feat_set in feature_sets.items():
        data_X_feature_set = data_X[feat_set]
        logger.info('Training models using feature subset: ' + feat_set_name)
        feature_set_output_dir = output_dir.joinpath(feat_set_name)
        for model_name, model in models.items():
            if model_name in ['en', 'lasso', 'lr']:
                # Use multitarget classification helper if the model is not
                # inherently multioutput
                model = MultiOutputClassifier(model)
            training_handler = handlers.ModelTraining(model)

            if validate is True:
                X_valid_feature_set = X_valid[feat_set]
                kfold_handler = kfold.RepeatedKFold(
                    n_iter=n_iter, n_folds=n_folds,
                    data_X=data_X_feature_set, data_Y=data_Y,
                    training_handler=training_handler,
                    X_valid=X_valid_feature_set, Y_valid=Y_valid,
                    output_type='classification')
            else:
                kfold_handler = kfold.RepeatedKFold(
                    n_iter=n_iter,
                    n_folds=n_folds,
                    training_handler=training_handler,
                    data_X=data_X_feature_set, data_Y=data_Y,
                    output_type='classification')
            logger.info('Training multitask approach using model: ' + model_name)
            model_results = kfold_handler.repeated_kfold(
                training_args=train_args, resampler=resampler)
            logger.info('Finished training for: ' + model_name)

            model_output_dir = feature_set_output_dir.joinpath(
                model_name + '/')
            os.makedirs(model_output_dir, exist_ok=True)
            write_results(model_results, model_output_dir)


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    main(args)

