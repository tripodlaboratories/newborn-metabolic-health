"""Script for running deep multi-task experiments."""
import argparse
import logging
import os
from pathlib import PurePath

import pandas as pd
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning.loggers as pl_loggers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import optim, Tensor
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss

from biobank_project.deep_mtl.training import handlers, kfold, utils
from biobank_project.deep_mtl.models import ensemble, multicategory
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
    parser.add_argument(
        '--num_workers', type=int, help='explicitly specified number of workers',
        default=10)
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
    input_file = args.input_file
    output_dir = PurePath(args.output_dir)
    tasks = args.tasks
    n_iter = args.n_iter
    validate = args.validate
    num_workers = args.num_workers

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

    # Split data into X and Y
    data_X = input_data.drop(outcomes, axis=1)
    data_Y = input_data[outcomes]
    gestage = included_metadata.loc[data_X.index, ['gacat']]
    if validate is True:
        data_X, X_valid, data_Y, Y_valid, gestage, ga_valid = train_test_split(
            data_X, data_Y, gestage, random_state=101)

    X_train, X_test, Y_train, Y_test, ga_train, ga_test = train_test_split(
        data_X, data_Y, gestage, random_state=101)

    # Train with different models
    # seed_everything(101)
    n_features = len(data_X.columns)
    n_tasks = len(outcomes)
    base_model_args = dict(n_features=n_features, n_hidden=100, n_tasks=n_tasks)
    pos_weight = Tensor(Y_train.apply(utils.get_pos_weight))
    model = multicategory.MultiCategoryEnsemble(
        categories=sorted(gestage['gacat'].unique()),
        base_model=ensemble.EnsembleNetwork,
        base_model_args=base_model_args,
        simple_average_combine=True,
        learn_model_weights=False,
        pos_weight_for_loss=pos_weight
    )

    # Set up data loaders
    batch_size = 3000
    scaler = StandardScaler()
    train_index = X_train.index.tolist()
    assert sorted(train_index) == sorted(Y_train.index.tolist()) == sorted(ga_train.index.tolist())
    train_dataloader = DataLoader(
        multicategory.MultiCategoryDataset(
            (Tensor(scaler.fit_transform(X_train.values)), Tensor(Y_train.values), ga_train.values.ravel()),
            reference_index=train_index), batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_index = X_test.index.tolist()
    assert sorted(test_index) == sorted(Y_test.index.tolist()) == sorted(ga_test.index.tolist())
    val_dataloader = DataLoader(
        multicategory.MultiCategoryDataset(
            (Tensor(scaler.transform(X_test.values)), Tensor(Y_test.values), ga_test.values.ravel()),
            reference_index=test_index), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if validate is True:
        valid_index = X_valid.index.tolist()
        assert sorted(valid_index) == sorted(Y_valid.index.tolist()) == sorted(ga_valid.index.tolist())
        holdout_test_dataloader = DataLoader(
            multicategory.MultiCategoryDataset(
                (Tensor(scaler.transform(X_valid.values)), Tensor(Y_valid.values), ga_valid.values.ravel()),
                reference_index=valid_index), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Set up logging/tensorboard
    os.makedirs(output_dir, exist_ok=True)
    tb_logger = pl_loggers.TensorBoardLogger(
        name='gestage_ensemble',
        save_dir=output_dir.joinpath('logs/')
    )

    # Set up model training
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=5)
    trainer = Trainer(
        max_epochs=500,
        callbacks=[early_stop_callback],
        logger=tb_logger,
        log_every_n_steps=10
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, holdout_test_dataloader)


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    main(args)
