"""Script for running deep multi-task experiments."""
import argparse
import logging
import os
from pathlib import PurePath

import pandas as pd
from sklearn.model_selection import train_test_split
from torch import optim
from torch.nn import BCEWithLogitsLoss
import wandb
import yaml

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
        '-t', '--tasks', type=str, default=None,
        help='Text file with tasks of interest, one task per line.')
    parser.add_argument(
        '--column_specification', type=str, default=None,
        help="Column specification YML containing keys: 'id', 'features', 'outcomes'")
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
        '--bottleneck_sequence', type=str, default='1,2,3,4,5,10,20',
        help='comma-separated sequence of bottleneck units to use.')
    parser.add_argument(
        '--drop_sparse', action='store_true', help='Drop sparse columns from input data.')
    # Optional wandb logging
    parser.add_argument(
        '--use_wandb', action='store_true', help="Enable Weights & Biases logging")
    parser.add_argument(
        '--experiment_name', type=str, default='deep_mtl_bottleneck_models',
        help='Name for the experiment group in W&B'
    )
    return parser


def read_lines(file) -> list:
    with open(file) as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


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
    col_spec_file = args.column_specification
    n_iter = args.n_iter
    cases_only = args.cases_only
    single_condition = args.single_condition
    validate = args.validate
    drop_sparse = args.drop_sparse
    use_wandb = args.use_wandb
    wandb_experiment_name = args.experiment_name

    # Handle features and outcomes
    if tasks is not None:
        features = None
        outcomes = read_lines(tasks)
        id_col = 'row_id'
    elif col_spec_file is not None:
        with open(col_spec_file, 'r') as f:
            col_spec  = yaml.safe_load(f)
        features = col_spec['features']
        outcomes = col_spec['outcomes']
        id_col = col_spec['id']
    else:
        raise ValueError('Must provide one of the following options: --tasks OR --column_specification')

    # Read in data
    input_data = pd.read_csv(input_file, low_memory=False)
    input_data.set_index(id_col, inplace=True)
    metadata = pd.read_csv('./data/processed/metadata.csv', low_memory=False)
    metadata.set_index(id_col, inplace=True)

    # Model-specific args
    bottleneck_sequence = [int(i) for i in args.bottleneck_sequence.split(',')]

    # Subset data based on gestational ages used
    included_ga_range = read_lines('./config/gestational_age_ranges.txt')
    included_metadata = metadata[metadata['gacat'].isin(included_ga_range)]
    input_data = input_data.loc[included_metadata.index, :]
    assert sorted(input_data.index) == sorted(included_metadata.index)

    # Handling for cases only
    if cases_only is True:
        input_data['total_conditions'] = input_data[outcomes].sum(axis=1)
        data_subset = input_data[input_data['total_conditions'] >= 1].drop(
            columns=['total_conditions'])
        if data_subset.shape[0] >= input_data.shape[0]:
            logger.warning('--cases_only flag did not reduce rows of data.')
        input_data = data_subset
        assert 'total_conditions' not in input_data

    if single_condition is True:
        input_data['total_conditions'] = input_data[outcomes].sum(axis=1)
        data_subset = input_data[input_data['total_conditions'] == 1].drop(
            column=['total_conditions'])
        if data_subset.shape[0] >= input_data.shape[0]:
            logger.warning('--single_condition flag did not reduce rows of data.')
        input_data = data_subset
        assert 'total_conditions' not in input_data

    # Drop sparse columns
    if drop_sparse:
        dropped_input = input_data.dropna(thresh=len(input_data) / 2, axis=1).copy()
        dropped_input.dropna(inplace=True)
        dropped_cols = set(input_data.columns).difference(dropped_input.columns)
        logger.info(f'Dropped columns from --drop_sparse option {dropped_cols}')
        input_data = dropped_input.copy()

    # Split data into X and Y
    if features is None:
        data_X = input_data.drop(outcomes, axis=1)
        data_Y = input_data[outcomes]
    else:
        common_features = [col for col in input_data if col in features]
        if len(common_features) != len(features):
            logger.warning((
                'Number of input features do not match feature specification:'
                f'Using {len(common_features)} common features.'
        ))
        data_X = input_data[common_features]
        data_Y = input_data[outcomes]

    if validate is True:
        logger.info('Setting up experiment to use holdout validation.')
        data_X, X_valid, data_Y, Y_valid = train_test_split(data_X, data_Y, random_state=101)

    # Train with different models
    utils.seed_torch(101)
    n_features = len(data_X.columns)
    n_tasks = len(outcomes)
    n_hidden = 100

    all_models = {}
    for n_bottleneck in bottleneck_sequence:
        bottle_spec = '_bottle_' + str(n_bottleneck)
        bottleneck_models = {
            'multi_output' + bottle_spec: bottleneck.ThreeLayerBottleneck(
                n_features=n_features, n_outputs=n_tasks,
                n_hidden=n_hidden, n_bottleneck=n_bottleneck),
            'ensemble' + bottle_spec: bottleneck.EnsembleNetwork(
                n_features=n_features, n_tasks=n_tasks,
                n_hidden=n_hidden, n_bottleneck=n_bottleneck),
            'parallel_ensemble' + bottle_spec: bottleneck.ParallelEnsembleNetwork(
                n_features=n_features, n_tasks=n_tasks,
                n_hidden=n_hidden, n_bottleneck=n_bottleneck),
            'large_multi_output' + bottle_spec: bottleneck.ThreeLayerBottleneck(
                n_features=n_features, n_outputs=n_tasks,
                n_bottleneck=n_bottleneck,
                hidden_layer_spec={
                    'hidden_1': n_hidden * n_tasks,
                    'hidden_2': n_hidden * n_tasks,
                    'hidden_3': n_hidden * n_tasks
                    }),
        }
        all_models = dict(all_models, **bottleneck_models)

    # Set up model training
    batch_size = 3000
    shuffle_batch = True
    n_epochs = 50
    early_stopping_patience = 5
    early_stopping_handler = handlers.EarlyStopping(
        patience=early_stopping_patience)
    resampler = MajorityDownsampler(random_state=101)

    os.makedirs(output_dir, exist_ok=True)
    for model_name, model in all_models.items():
        # Weights and biases setup
        if use_wandb:
            run = wandb.init(
                project='deep-metabolic-health-index',
                name=f"{model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
                job_type='training',
                tags=[model_name],
                group=wandb_experiment_name,
                config={
                    "model_type": model_name,
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "n_features": n_features,
                    "n_tasks": n_tasks,
                    "n_iter": n_iter,
                    "cases_only": cases_only,
                },
                reinit=True)
            wandb.watch(model, log="all", log_freq=25)
        else:
            run = wandb.init(mode='disabled')

        training_handler = handlers.BottleneckModelTraining(
            model=model, batch_size=batch_size, shuffle_batch=shuffle_batch,
            optimizer_class=optim.Adam, wandb_run=run)
        train_args = {
            'n_epochs': n_epochs,
            'criterion': BCEWithLogitsLoss(reduction='mean'),
            'colnames': data_Y.columns,
            'early_stopping_handler': early_stopping_handler
            }
        if validate is True:
            kfold_handler = kfold.RepeatedKFold(
                n_iter=n_iter, n_folds=5, data_X=data_X, data_Y=data_Y,
                training_handler=training_handler, X_valid=X_valid, Y_valid=Y_valid)
        else:
            kfold_handler = kfold.RepeatedKFold(
                n_iter=n_iter, n_folds=10, data_X=data_X, data_Y=data_Y,
                training_handler=training_handler)

        model_results = kfold_handler.repeated_kfold(
            training_args=train_args, resampler=resampler, class_tasks=outcomes)
        logger.info('Finished model training for: ' + model_name)

        # Write out results
        model_output_dir = output_dir.joinpath(model_name + '/')
        os.makedirs(model_output_dir, exist_ok=True)
        write_results(model_results, model_output_dir)
        logger.info('Model results written to: ' + str(model_output_dir) + '/')

        # wandb specific cleanup
        # TODO: I think calling run.finish() in disabled mode is causing the following error:
        # "terminate called without an active exception"
        if use_wandb:
            run.finish()
        else:
            # In disabled mode, run.finish() causes "terminate called without active exception"
            del run


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    main(args)
