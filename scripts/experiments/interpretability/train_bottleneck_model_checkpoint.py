"""Script for creating a bottlenecked model checkpoint - following as much of the logic as run_deep_mtl_bottleneck.py"""
import argparse
import json
import logging
import os
from pathlib import PurePath

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import optim, Tensor
from torch.nn import BCEWithLogitsLoss
import wandb
import yaml

from biobank_project.deep_mtl.training import handlers, kfold, utils
from biobank_project.deep_mtl.training.schedulers import get_scheduler_creator
from biobank_project.deep_mtl.models import bottleneck
from biobank_project.deep_mtl.sampling import MajorityDownsampler


def get_args():
    parser = argparse.ArgumentParser(
        description='Create a checkpointed model for the 1-unit bottleneck.',
        conflict_handler='resolve')
    parser.add_argument(
        '-i', '--input_data', type=str, default=None,
        help='input data for model training.')
    parser.add_argument(
        '-o', '--output', type=str, help='output directory for results and model checkpoint',
        required=True, metavar='OUTPUT_DIR', dest='output_dir')
    parser.add_argument(
        '-t', '--tasks', type=str, default=None,
        help='Text file with tasks of interest, one task per line.')
    parser.add_argument(
        '--column_specification', type=str, default=None,
        help="Column specification YML containing keys: 'id', 'features', 'outcomes'")
    parser.add_argument(
        '--metadata', type=str, default=None,
        help='input metadata for gestational age ranges')
    parser.add_argument(
        '--gestational_age_file', default=None,
        help='gestational age ranges to use')
    parser.add_argument(
        '--drop_sparse', action='store_true', default=False,
        help='Drop sparse columns in the input data before attempting model training')
    parser.add_argument(
        '--imbalance_strategy', type=str, default='majority_downsampler',
        help='Imbalanced data strategy, one of either "majority_downsampler" or "loss_pos_weight')
    parser.add_argument(
        '--pos_weight_attenuation', type=float, default=None,
        help="Attenuation factor for pos_weight imbalance strategy: e.g., pos_weight = pos_weight / attenuation_factor only when greater than 1.0")
    parser.add_argument(
        '--bottleneck', type=int, default=1,
        help='Number of bottleneck units to use.')
    parser.add_argument(
        '--lr_scheduler', type=str, default=None,
        help='Specify a LR Scheduler class.')
    return parser.parse_args()


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
    logger = logging.getLogger('Train Model Checkpoint')

    input_file = args.input_data
    output_dir = PurePath(args.output_dir)
    tasks = args.tasks
    col_spec_file = args.column_specification
    drop_sparse = args.drop_sparse
    imbalance_strategy = args.imbalance_strategy
    scheduler_name = args.lr_scheduler
    pos_weight_attenuation = args.pos_weight_attenuation
    n_bottleneck = args.bottleneck

    # Handle features and outcomes
    if tasks is not None:
        features = None
        outcomes = read_lines(tasks)
        id_col = 'row_id'
    elif col_spec_file is not None:
        with open(col_spec_file, 'r') as f:
            col_spec = yaml.safe_load(f)
        features = col_spec['features']
        outcomes = col_spec['outcomes']
        id_col = col_spec['id']
    else:
        raise ValueError('Must provide one of the following options: --tasks OR --oclumn_spec')

    # Read in data
    input_data = pd.read_csv(input_file, low_memory=False)
    input_data.set_index(id_col, inplace=True)
    metadata = pd.read_csv('./data/processed/metadata.csv', low_memory=False)
    metadata.set_index(id_col, inplace=True)
    included_ga_range = read_lines('./config/gestational_age_ranges.txt')
    included_metadata = metadata[metadata['gacat'].isin(included_ga_range)]
    input_data = input_data.loc[included_metadata.index, :]
    assert sorted(input_data.index) == sorted(included_metadata.index)

    if scheduler_name is not None:
        scheduler_creator_fn = get_scheduler_creator(scheduler_name)

    if drop_sparse:
        dropped_input = input_data.dropna(thresh=len(input_data) / 2, axis=1).copy()
        dropped_input.dropna(inplace=True)
        dropped_cols = set(input_data.columns).difference(dropped_input.columns)
        logger.info(f'Dropped columns from --drop_sparse option {dropped_cols}')
        input_data = dropped_input.copy()

    # Split data into X and Y
    scaler = StandardScaler()
    if features is None:
        data_X = input_data.drop(outcomes, axis=1, errors='ignore')
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

    # Preprocess before training
    # NOTE: Scaler only should be applied to metabolite features.
    if 'categorical_features' in col_spec.keys():
        categorical_features = col_spec.get('categorical_features')
        features_to_transform = data_X.columns.difference(categorical_features)
        logger.info(f'Explicitly not scaling the following categorical features: {categorical_features}')
        data_X_scaled = data_X.copy()
        data_X_scaled[features_to_transform] = scaler.fit_transform(
            data_X_scaled[features_to_transform])
    else:
        data_X_scaled = scaler.fit_transform(data_X)

    data_X = pd.DataFrame(
        data_X_scaled,
        columns=data_X.columns,
        index=data_X.index)

    # Train
    utils.seed_torch(101)
    n_features = len(data_X.columns)
    n_tasks = len(outcomes)
    n_hidden = 100

    model = bottleneck.EnsembleNetwork(
        n_features=n_features, n_tasks=n_tasks,
        n_hidden=n_hidden, n_bottleneck=n_bottleneck)

    # Set up model training
    batch_size = 3000
    shuffle_batch = True
    n_epochs = 50
    early_stopping_patience = 5
    early_stopping_handler = handlers.EarlyStopping(
        patience=early_stopping_patience)

    # Options for downsampling or positive weight for data imbalance
    known_imbalance_strategies = ['majority_downsampler', 'loss_pos_weight']
    if imbalance_strategy not in known_imbalance_strategies:
        logger.warning(f"--imbalance_strategy must be one of {known_imbalance_strategies}, defaulting to None.")
        resampler = None
        pos_weight = None
    elif imbalance_strategy == 'majority_downsampler':
        resampler = MajorityDownsampler(random_state=101)
        pos_weight = None
    elif imbalance_strategy == 'loss_pos_weight':
        resampler = None
        pos_weight = data_Y.apply(utils.get_pos_weight).values
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

        if pos_weight_attenuation is not None:
            if pos_weight_attenuation < 1.0:
                raise ValueError('pos_weight_attenutation must be greater than 1.0, since used as pos_weight / pos_weight_attenuation')
            pos_weight = torch.where(
                pos_weight / pos_weight_attenuation > 1.0,
                pos_weight / pos_weight_attenuation,
                torch.tensor([1.0], dtype=torch.float32)
            )
        logger.info(f'pos_weight for loss: {pos_weight} across {data_Y.columns}')
    else:
        raise ValueError("Argument --imbalance_strategy unknown and/or not handled correctly.")

    os.makedirs(output_dir, exist_ok=True)
    # Write out results
    # Set up the handler for the metabolic health index model
    run = wandb.init(mode='disabled')
    if scheduler_name is not None:
        training_handler = handlers.BottleneckModelTraining(
            model=model,
            batch_size=batch_size, shuffle_batch=shuffle_batch,
            optimizer_class=torch.optim.Adam,
            scheduler_creator=lambda optimizer: scheduler_creator_fn(optimizer, config={}),
            wandb_run=run)
    else:
        training_handler = handlers.BottleneckModelTraining(
            model=model,
            batch_size=batch_size, shuffle_batch=shuffle_batch,
            optimizer_class=torch.optim.Adam,
            wandb_run=run)

    # Set up data for training and sample for test, evaluating when model
    # stopping should happen
    test_index = data_X.sample(1000, random_state=10).index
    X_train = data_X.loc[~data_X.index.isin(test_index)]
    Y_train = data_Y.loc[~data_X.index.isin(test_index)]
    X_test = data_X.loc[data_X.index.isin(test_index)]
    Y_test = data_Y.loc[data_Y.index.isin(test_index)]
    training_handler.set_training_data(X_train, Y_train)
    training_handler.set_test_data(X_test, Y_test)

    train_args = {
        'n_epochs': n_epochs,
        'criterion': BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight),
        'colnames': data_Y.columns,
        'early_stopping_handler': early_stopping_handler,
        'output_training_preds': True
        }

    results = training_handler.train(**train_args)
    write_results(results, output_dir)

    # Write out the outputs from the final model
    feature_tensor = torch.tensor(data_X.values).float()
    model_output = training_handler.model(
        feature_tensor, return_bottleneck=True)
    health_index_score = model_output[1]
    health_index_df = pd.DataFrame(
        health_index_score.detach().numpy(),
        index=data_X.index, columns=['health_index'])
    health_index_df.to_csv(
        str(output_dir.joinpath('health_index_output.csv')))

    # Write out the test set predictions and true values
    X_test.to_csv(str(output_dir.joinpath('X_test.csv')))
    Y_test.to_csv(str(output_dir.joinpath('Y_test.csv')))

    # Save the model using torch.save (or save the model weights?)
    torch.save(
        training_handler.model,
        str(output_dir.joinpath('checkpointed_model.pt')))
    torch.save(
        training_handler.model.state_dict(),
        str(output_dir.joinpath('model_state_dict.pt')))
    torch.save(
        training_handler.optimizer.state_dict(),
        str(output_dir.joinpath('optimizer_state_dict.pt')))
    joblib.dump(
        scaler,
        str(output_dir.joinpath('data_scaler.joblib')))

    # Delete no-op wandb run object
    del run


if __name__ == '__main__':
    args = get_args()
    main(args)