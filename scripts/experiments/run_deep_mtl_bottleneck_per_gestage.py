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
        '--cases_only', action='store_true',
        help='perform model training on cases only, ignoring controls.')
    parser.add_argument(
        '--single_condition', action='store_true',
        help='perform model training on cases with only one condition')
    parser.add_argument(
        '-v', '--validate', action='store_true',
        help='split data in train/test/validate.')
    # Optional wandb logging
    parser.add_argument(
        '--use_wandb', action='store_true', help="Enable Weights & Biases logging")
    parser.add_argument(
        '--experiment_name', type=str, default='bottleneck_models_per_gestage',
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
    n_iter = args.n_iter
    cases_only = args.cases_only
    single_condition = args.single_condition
    validate = args.validate
    use_wandb = args.use_wandb
    wandb_experiment_name = args.experiment_name

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

    # Train with different models
    utils.seed_torch(101)
    n_features = len(input_data.drop(outcomes, axis=1).columns)
    n_tasks = len(outcomes)
    n_hidden = 100

    bottleneck_sequence = [1, 2, 3, 4, 5, 10, 20]
    all_models = {}
    for n_bottleneck in bottleneck_sequence:
        bottle_spec = '_bottle_' + str(n_bottleneck)
        bottleneck_models = {
            'ensemble' + bottle_spec: bottleneck.EnsembleNetwork(
                n_features=n_features, n_tasks=n_tasks,
                n_hidden=n_hidden, n_bottleneck=n_bottleneck),
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
    # Train models specific to each gestational age
    for ga in included_ga_range:
        ga_output_dir = output_dir.joinpath(ga + '/')
        ga_included_metadata = included_metadata[included_metadata['gacat'] == ga]
        ga_input_data = input_data.loc[ga_included_metadata.index, :]
        assert sorted(ga_input_data.index) == sorted(ga_included_metadata.index)
        ga_input_data.dropna(inplace=True)

        # Split data into X and Y
        data_X = ga_input_data.drop(outcomes, axis=1)
        data_Y = ga_input_data[outcomes]

        if validate is True:
            logger.info('Using holdout validation for samples within gestational age range.')
            data_X, X_valid, data_Y, Y_valid = train_test_split(data_X, data_Y, random_state=101)

            # Append samples outside gestational age to the validation set
            samples_outside_ga_meta = included_metadata[included_metadata['gacat'] != ga]
            samples_outside_ga_idx = input_data.index.isin(samples_outside_ga_meta.index)
            samples_outside_ga = input_data.loc[samples_outside_ga_idx, :].copy()
            samples_outside_ga.dropna(inplace=True)
            X_valid = X_valid.append(samples_outside_ga.drop(outcomes, axis=1))
            Y_valid = Y_valid.append(samples_outside_ga[outcomes])
            X_valid = X_valid.sample(frac=1, random_state=101)
            Y_valid = Y_valid.loc[X_valid.index, :]

        logger.info('Constraining samples to the following gestational age: ' + ga)

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
                training_args=train_args, resampler=resampler)
            logger.info('Finished model training for: ' + model_name)

            # Write out results
            model_output_dir = ga_output_dir.joinpath(model_name + '/')
            os.makedirs(model_output_dir, exist_ok=True)
            write_results(model_results, model_output_dir)
            logger.info('Model results written to: ' + str(model_output_dir) + '/')
            run.finish()

if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    main(args)
