# TODO: Incorporate clinical and metabolic variables together in one model
"""Script for running deep multi-task experiments."""
import argparse
import logging
import os
from pathlib import PurePath
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import optim, Tensor
from torch.nn import BCEWithLogitsLoss

from biobank_project.deep_mtl.training import handlers, kfold, utils
from biobank_project.deep_mtl.models import bottleneck
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
        '-v', '--validate', action='store_true',
        help='split data in train/test/validate.')
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
    logger = logging.getLogger('deep_mtl_experiment')

    input_file = args.input_file
    output_dir = PurePath(args.output_dir)
    tasks = args.tasks
    n_iter = args.n_iter
    validate = args.validate

    # Read in data
    input_data = pd.read_csv(input_file, low_memory=False)
    input_data.set_index('row_id', inplace=True)
    metadata = pd.read_csv('./data/processed/metadata.csv', low_memory=False)
    metadata.set_index('row_id', inplace=True)
    outcomes = read_lines(tasks)

    # Subset data based on gestational ages used
    included_ga_range = read_lines('./config/gestational_age_ranges.txt')
    included_metadata = metadata[metadata['gacat'].isin(included_ga_range)].copy()
    input_data = input_data.loc[included_metadata.index, :]
    assert sorted(input_data.index) == sorted(included_metadata.index)

    # Drop sparse columns
    input_data.dropna(thresh=len(input_data) / 2, axis=1, inplace=True)

    # Set up clinical feature sets of interest
    included_metadata.rename(columns={'sex3': 'infant_sex'}, inplace=True)
    clinical_feature_sets = {
        'bwtga': ['gacat', 'bwtcat'],
        'minimal_vars': ['bwtcat', 'gacat', 'infant_sex'],
        'additional_risk_vars': ['gacat', 'bwtcat', 'infant_sex', 'ap1cat', 'ap5cat'],
        'apgar_only': ['ap1cat', 'ap5cat']
        'clinical_features_only': ['gacat', 'bwtcat', 'infant_sex', 'ap1cat', 'ap5cat']
    }
    # Collect all feature names for subsetting data
    all_clinical_features = [
        f for features in clinical_feature_sets.values() for f in features]
    all_clinical_features = list(set(all_clinical_features))
    all_metabolite_features = input_data.drop(outcomes, axis=1).columns
    all_metabolite_features = all_metabolite_features.to_list()
    all_features = all_metabolite_features + all_clinical_features
    # Merge input data with metadata columns of interest
    input_data = pd.merge(
        input_data, included_metadata[all_clinical_features],
        left_index=True, right_index=True, how='left')

    # Recode relevant metadata features
    # Recode infant sex encoding to {0,1} (in original data it is {1,2}
    # 0: males, 1: females
    n_males = input_data.infant_sex.value_counts().loc[1]
    input_data.replace({'infant_sex': {1: 0, 2: 1}}, inplace=True)
    n_males_after_recoding = input_data.infant_sex.value_counts().loc[0]
    assert len(input_data.infant_sex.unique()) == 2
    assert n_males == n_males_after_recoding

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

    input_data.dropna(inplace=True)

    # Split data into X and Y
    data_X = input_data[all_features]
    data_Y = input_data[outcomes]

    if validate is True:
        logger.info('Setting up experiment to use holdout validation.')
        data_X, X_valid, data_Y, Y_valid = train_test_split(data_X, data_Y, random_state=101)

    # Train with different models
    utils.seed_torch(101)
    n_tasks = len(outcomes)
    n_hidden = 100
    bottleneck_sequence = [1, 5, 10, 20]

    # Set up model training
    batch_size = 3000
    shuffle_batch = True
    n_epochs = 50
    early_stopping_patience = 5
    early_stopping_handler = handlers.EarlyStopping(
        patience=early_stopping_patience)
    resampler = MajorityDownsampler(random_state=101)
    os.makedirs(output_dir, exist_ok=True)

    # Include handling of different feature sets for data_X and data_Y
    # for different clinical data feature sets
    # Iterate over feature sets
    for feature_set_name, feature_set in clinical_feature_sets.items():
        if feature_set_name == 'clinical_features_only':
            logger.info(f'{feature_set_name}: {feature_set}')
            data_X_subset = data_X[feature_set]
        else:
            logger.info(f'Additional features: {feature_set_name}: {feature_set}')
            data_X_subset = data_X[all_metabolite_features + feature_set]
        feature_set_output_dir = output_dir.joinpath(feature_set_name + '/')
        os.makedirs(feature_set_output_dir, exist_ok=True)

        # Model definition is dependent on the number of features in the feature
        # subset
        n_features = len(data_X_subset.columns)
        all_models = {}
        for n_bottleneck in bottleneck_sequence:
            bottle_spec = '_bottle_' + str(n_bottleneck)
            bottleneck_models = {
                'ensemble' + bottle_spec: bottleneck.EnsembleNetwork(
                    n_features=n_features, n_tasks=n_tasks,
                    n_hidden=n_hidden, n_bottleneck=n_bottleneck)
            }
            all_models = dict(all_models, **bottleneck_models)

        for model_name, model in all_models.items():
            training_handler = handlers.BottleneckModelTraining(
                model=model, batch_size=batch_size, shuffle_batch=shuffle_batch,
                optimizer_class=optim.Adam)

            train_args = {
                'n_epochs': n_epochs,
                'criterion': BCEWithLogitsLoss(reduction='mean'),
                'colnames': data_Y.columns,
                'early_stopping_handler': early_stopping_handler
                }
            if validate is True:
                kfold_handler = kfold.RepeatedKFold(
                    n_iter=n_iter, n_folds=5,
                    data_X=data_X_subset, data_Y=data_Y,
                    training_handler=training_handler,
                    X_valid=X_valid[all_metabolite_features + feature_set],
                    Y_valid=Y_valid)
            else:
                kfold_handler = kfold.RepeatedKFold(
                    n_iter=n_iter, n_folds=10,
                    data_X=data_X_subset, data_Y=data_Y,
                    training_handler=training_handler)

            model_results = kfold_handler.repeated_kfold(
                training_args=train_args, resampler=resampler, class_tasks=outcomes)
            logger.info('Finished model training for: ' + model_name)

            # Write out results
            # Create feature set specific directory for results
            model_output_dir = feature_set_output_dir.joinpath(
                model_name + '/')
            os.makedirs(model_output_dir, exist_ok=True)
            write_results(model_results, model_output_dir)
            logger.info(f'Model results written to: {model_output_dir}/')


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    main(args)

