# Machine learning model comparison script
import argparse
import logging
import os
from pathlib import PurePath

import numpy as np
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from torch import optim, Tensor
from torch.nn import BCEWithLogitsLoss, MSELoss
from xgboost import XGBClassifier

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
    parser.add_argument(
        '--as_health_index', action='store_true',
        help='Recode sick and healthy infants in analogy to metabolic health index.'
    )
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
    as_health_index = args.as_health_index

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

    # Drop sparse columns
    input_data.dropna(thresh=len(input_data) / 2, axis=1, inplace=True)
    input_data.dropna(inplace=True)

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

    if as_health_index is True:
        logger.info('Recoding outcomes as one outcome of healthy/sick.')
        input_data['total_conditions'] = input_data[outcomes].sum(axis=1)
        input_data['healthy_infant'] = np.where(
            input_data['total_conditions'] == 0, 1, 0)
        input_data.drop(
            columns=outcomes + ['total_conditions'], inplace=True)
        outcomes = ['healthy_infant']

    # Split data into X and Y
    data_X = input_data.drop(outcomes, axis=1)
    data_Y = input_data[outcomes]

    if validate is True:
        logger.info('Setting up experiment to use holdout validation.')
        data_X, X_valid, data_Y, Y_valid = train_test_split(data_X, data_Y, random_state=101)

    # Set up single task classification and regression models from sklearn
    # choices.
    utils.seed_torch(101)

    # Setup model training for widespread model comparison
    # Model specific hyperparameters for logistic regression models
    l1_ratio_range = [0.1, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    num_C_vals = 20
    max_iter = 10000
    n_cv_folds = 5
    classification_scoring_metric='roc_auc'

    # Hyperparameter tuning for HGBC and XGBoost
    random_state_int = 101
    random_search_args = {
        'n_iter': 100,
        'cv': 3,
        'random_state': random_state_int,
        'n_jobs': n_jobs,
        'verbose': 1
    }
    n_estimators = [50, 100, 200, 400, 600, 800, 1000, 1400, 1800, 2000]

    # Hyperparamter Tuning for RandomForest
    max_depth = [10, 30, 50, 70, 90, 100, None]
    min_samples_split = [2, 5, 10]
    rf_grid = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
    }
    rf = RandomForestClassifier(n_jobs=1)
    rfcv = RandomizedSearchCV(
        rf, param_distributions=rf_grid, **random_search_args)

    # Hyperparameter tuning for hgbc
    histgb_grid = {
        'learning_rate': [0.001, 0.1, 0.5, 1.0],
        'max_iter': n_estimators,
        'l2_regularization': [0.0, 0.1, 1.0, 5.0, 10., 50.],
        'min_samples_leaf': [10, 20, 50, 100]
    }
    histgb = HistGradientBoostingClassifier(
        random_state=random_state_int, loss='binary_crossentropy')
    histgbcv = RandomizedSearchCV(
        histgb, param_distributions=histgb_grid, **random_search_args )

    # Hyperparameter tuning for xgboost
    learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    reg_lambda = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
    xgb_grid = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.1, 1., 3., 5., 7., 10.],
        'max_depth': [6, 10, 15, 20],
        'colsample_bytree': [0.4, 0.6, 0.8, 1.0],
        'gamma': [0.0, 0.25, 0.5, 1.0],
        'reg_lambda': reg_lambda,
    }
    xgb = XGBClassifier(verbosity=0, n_jobs=1, use_label_encoder=False)
    xgbcv = RandomizedSearchCV(
        xgb, param_distributions=xgb_grid, **random_search_args)

    # Model specific hyperparameters for ensemble methods
    models = {
        'rf': rfcv,
        'hgbc': histgbcv,
        'xgboost': xgbcv,
        'en': LogisticRegressionCV(
            penalty='elasticnet', cv=n_cv_folds,
            scoring=classification_scoring_metric,
            l1_ratios=l1_ratio_range, solver='saga',
            Cs=num_C_vals, n_jobs=n_jobs, max_iter=max_iter),
        'lasso': LogisticRegressionCV(
            penalty='l1', solver='saga', cv=n_cv_folds,
            scoring=classification_scoring_metric,
            Cs=num_C_vals, n_jobs=n_jobs, max_iter=max_iter)
    }

    # Set up model training
    n_folds = 5
    resampler = MajorityDownsampler(random_state=101)

    train_args = {'colnames': data_Y.columns}

    for model_name, model in models.items():
        if (as_health_index is False) and (model_name in ['en', 'lasso', 'hgbc', 'xgboost']):
            # Use multitarget classification helper if the model is not
            # inherently multioutput
            model = MultiOutputClassifier(model)
        training_handler = handlers.ModelTraining(model)

        if validate is True:
            kfold_handler = kfold.RepeatedKFold(
                n_iter=n_iter, n_folds=n_folds, data_X=data_X, data_Y=data_Y,
                training_handler=training_handler, X_valid=X_valid,
                Y_valid=Y_valid, output_type='classification')
        else:
            kfold_handler = kfold.RepeatedKFold(
                n_iter=n_iter,
                n_folds=n_folds,
                training_handler=training_handler,
                data_X=data_X, data_Y=data_Y,
                output_type='classification')
        logger.info('Training multitask approach using model: ' + model_name)
        model_results = kfold_handler.repeated_kfold(
            training_args=train_args, resampler=resampler)
        logger.info('Finished training for: ' + model_name)

        model_output_dir = output_dir.joinpath(
            model_name + '/')
        os.makedirs(model_output_dir, exist_ok=True)
        write_results(model_results, model_output_dir)


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    main(args)

