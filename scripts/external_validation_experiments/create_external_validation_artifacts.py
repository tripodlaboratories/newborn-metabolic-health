"""Script for creating a bottlenecked model checkpoint."""
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

from biobank_project.deep_mtl.training import handlers, kfold, utils
from biobank_project.deep_mtl.models import bottleneck
from biobank_project.deep_mtl.sampling import MajorityDownsampler


def get_argparser():
    parser = argparse.ArgumentParser(
        description='Create a checkpointed model for the 1-unit bottleneck.',
        conflict_handler='resolve')
    parser.add_argument(
        '-i', '--input_data', type=str, default=None,
        help='input data for model training.')
    parser.add_argument(
        '-c', '--config', type=str, default=None,
        help='Config file for generating external validation artifacts')
    parser.add_argument(
        '--outcomes_file', type=str, default=None,
        help='provide separate outcomes data for model training')
    parser.add_argument(
        '--metadata', type=str, default=None,
        help='input metadata for gestational age ranges')
    parser.add_argument(
        '--gestational_age_file', default=None,
        help='gestational age ranges to use')
    parser.add_argument(
        '-o', '--output', type=str, help='output directory for results and model checkpoint',
        required=True, metavar='OUTPUT_DIR', dest='output_dir')
    parser.add_argument(
        '--drop_sparse_cols', action='store_true', default=False,
        help='Drop sparse columns in the input data before attempting model training')
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
    logger = logging.getLogger('Model Training')

    input_data = args.input_data
    config_file = args.config
    outcomes_file = args.outcomes_file
    drop_sparse_cols = args.drop_sparse_cols
    output_dir = PurePath(args.output_dir)
    tasks = './config/neonatal_covariates.txt'
    gestational_age_file = args.gestational_age_file

    if config_file is not None:
        with open(config_file) as f:
            config = json.load(f)
        input_data_file = config.get('input')
        input_data = pd.read_csv(input_data_file, low_memory=False)
        input_data.set_index('row_id', inplace=True)
        metadata_file = config.get('metadata')
        included_ga_range = config.get('gestational_age_range')
        outcomes = config.get('outcomes')
        metabolite_subset = config.get('metabolites')

        metadata = pd.read_csv(metadata_file, low_memory=False)
        metadata.set_index('row_id', inplace=True)
        included_metadata = metadata[metadata['gacat'].isin(included_ga_range)]
        input_data = input_data.loc[included_metadata.index, :]
        assert sorted(input_data.index) == sorted(included_metadata.index)

    else:
        # Read in data
        input_data = pd.read_csv(input_data, low_memory=False)
        input_data.set_index('row_id', inplace=True)
        outcomes = read_lines(tasks)
        metabolite_subset = None

        # Subset data based on gestational ages used
        if gestational_age_file is not None:
            metadata = pd.read_csv(args.metadata, low_memory=False)
            metadata.set_index('row_id', inplace=True)
            included_ga_range = read_lines(gestational_age_file)
            included_metadata = metadata[metadata['gacat'].isin(included_ga_range)]
            input_data = input_data.loc[included_metadata.index, :]
            assert sorted(input_data.index) == sorted(included_metadata.index)


    # Split data into X and Y
    scaler = StandardScaler()

    if outcomes_file is None:
        if metabolite_subset is not None:
            data_X = input_data[metabolite_subset]
        # Drop sparse columns
        if drop_sparse_cols is True:
            data_X.dropna(thresh=len(input_data) / 2, axis=1, inplace=True)
            data_X.dropna(inplace=True)

        data_X = data_X.drop(outcomes, axis=1, errors='ignore')
        data_Y = input_data[outcomes]
    else:
        data_Y = pd.read_csv(args.outcomes_file).set_index('row_id')
        data_Y = data_Y[outcomes]
        merged = pd.merge(
            input_data, data_Y, left_index=True, right_index=True, how='inner')
        data_X = merged.drop(outcomes, axis=1)

        if metabolite_subset is not None:
            data_X = input_data[metabolite_subset]
        # Drop sparse columns
        if drop_sparse_cols is True:
            data_X.dropna(thresh=len(input_data) / 2, axis=1, inplace=True)
            data_X.dropna(inplace=True)

        data_Y = merged[outcomes]
        assert sorted(data_X.index) == sorted(data_Y.index)

    # Preprocess before training
    data_X = pd.DataFrame(
        scaler.fit_transform(data_X),
        columns=data_X.columns,
        index=data_X.index)

    # Train
    utils.seed_torch(101)
    n_features = len(data_X.columns)
    n_tasks = len(outcomes)
    n_hidden = 100
    n_bottleneck = 1

    metabolic_health_model = bottleneck.EnsembleNetwork(
        n_features=n_features, n_tasks=n_tasks,
        n_hidden=n_hidden, n_bottleneck=n_bottleneck)

    # Set up model training
    batch_size = 3000
    shuffle_batch = True
    n_epochs = 50
    early_stopping_patience = 5
    early_stopping_handler = handlers.EarlyStopping(
        patience=early_stopping_patience)

    os.makedirs(output_dir, exist_ok=True)
    model_name = 'metabolic_health_index'

    # Write out results
    model_output_dir = output_dir.joinpath(model_name + '/')
    os.makedirs(model_output_dir, exist_ok=True)

    # Set up the handler for the metabolic health index model
    training_handler = handlers.BottleneckModelTraining(
        model=metabolic_health_model, batch_size=batch_size,
        shuffle_batch=shuffle_batch,
        optimizer_class=optim.Adam)

    # Set up data for training and dummy sample for test - we don't really care
    # about the evaluations on the test data, the handler just requires it.
    training_handler.set_training_data(data_X, data_Y)
    training_handler.set_test_data(
        data_X.sample(10, random_state=10),
        data_Y.sample(10, random_state=10))

    train_args = {
        'n_epochs': n_epochs,
        'criterion': BCEWithLogitsLoss(reduction='mean'),
        'colnames': data_Y.columns,
        'early_stopping_handler': early_stopping_handler,
        'output_training_preds': True
        }

    results = training_handler.train(**train_args)
    write_results(results, model_output_dir)

    # Write out the outputs from the final model
    feature_tensor = torch.tensor(data_X.values).float()
    model_output = training_handler.model(
        feature_tensor, return_bottleneck=True)
    health_index = model_output[1]
    health_index_df = pd.DataFrame(
        health_index.detach().numpy(),
        index=data_X.index, columns=['health_index'])
    health_index_df.to_csv(
        str(model_output_dir.joinpath('health_index_output.csv')))

    # Save the model using torch.save (or save the model weights?)
    torch.save(
        training_handler.model,
        str(model_output_dir.joinpath('checkpointed_model.pt')))
    torch.save(
        training_handler.model.state_dict(),
        str(model_output_dir.joinpath('model_state_dict.pt')))
    torch.save(
        training_handler.optimizer.state_dict(),
        str(model_output_dir.joinpath('optimizer_state_dict.pt')))
    joblib.dump(
        scaler,
        str(model_output_dir.joinpath('data_scaler.joblib')))


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    main(args)
