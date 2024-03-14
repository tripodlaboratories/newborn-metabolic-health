"""Script for running deep multi-task experiments."""
import argparse
import logging
import os
from pathlib import PurePath

import pandas as pd
from sklearn.model_selection import train_test_split
from torch import optim, Tensor
from torch.nn import BCEWithLogitsLoss

from biobank_project.deep_mtl.training import handlers, kfold, utils
from biobank_project.deep_mtl.models import base, ensemble
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
    cases_only = args.cases_only
    validate = args.validate

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

    # Split data into X and Y
    data_X = input_data.drop(outcomes, axis=1)
    data_Y = input_data[outcomes]

    if validate is True:
        logger.info('Setting up experiment to use holdout validation.')
        data_X, X_valid, data_Y, Y_valid = train_test_split(data_X, data_Y, random_state=101)

    # Train with different models
    utils.seed_torch(101)
    n_features = len(data_X.columns)
    n_tasks = len(outcomes)
    all_models = {
        'multi_output': base.ThreeLayerMultiOutput(
            n_features=n_features, n_outputs=n_tasks,
            hidden_layer_spec={
                'hidden_1': 100,
                'hidden_2': 100,
                'hidden_3': 100}),
        'large_multi_output': base.ThreeLayerMultiOutput(
            n_features=n_features, n_outputs=n_tasks,
            hidden_layer_spec={
                'hidden_1': 400,
                'hidden_2': 400,
                'hidden_3': 400}),
        'ensemble': ensemble.EnsembleNetwork(
            n_features=n_features, n_hidden=100, n_tasks=n_tasks),
        'parallel_ensemble': ensemble.ParallelEnsembleNetwork(
            n_features=n_features, n_hidden=100, n_tasks=n_tasks),
    }

    # Set up model training
    batch_size = 3000
    shuffle_batch = True
    n_epochs = 50
    pos_weight = Tensor(data_Y.apply(utils.get_pos_weight))
    early_stopping_patience = 5
    early_stopping_handler = handlers.EarlyStopping(
        patience=early_stopping_patience)
    # resampler = MajorityDownsampler(random_state=101)
    resampler = None
    results = {k: None for k in all_models.keys()}

    os.makedirs(output_dir, exist_ok=True)
    for model_name, model in all_models.items():
        training_handler = handlers.ModelTraining(
            model=model, batch_size=batch_size, shuffle_batch=shuffle_batch,
            optimizer_class=optim.Adam)

        train_args = {
            'n_epochs': n_epochs,
            'criterion': BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight),
            'colnames': data_Y.columns,
            'early_stopping_handler': early_stopping_handler,
            'output_training_preds': True
            }
        kfold_handler = kfold.RepeatedKFold(
            n_iter=n_iter, n_folds=5, data_X=data_X, data_Y=data_Y,
            training_handler=training_handler)

        if validate is True:
            kfold_handler = kfold.RepeatedKFold(
                n_iter=n_iter, n_folds=5, data_X=data_X, data_Y=data_Y,
                training_handler=training_handler, X_valid=X_valid, Y_valid=Y_valid)
        else:
            kfold_handler = kfold.RepeatedKFold(
                n_iter=n_iter, n_folds=5, data_X=data_X, data_Y=data_Y,
                training_handler=training_handler)

        model_results = kfold_handler.repeated_kfold(
            training_args=train_args, resampler=resampler)
        logger.info('Finished model training for: ' + model_name)

        # Write out results
        model_output_dir = output_dir.joinpath(model_name + '/')
        os.makedirs(model_output_dir, exist_ok=True)
        write_results(model_results, model_output_dir)
        logger.info('Model results written to: ' + str(model_output_dir) + '/')


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    main(args)
