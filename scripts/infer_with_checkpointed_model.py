"""Use a checkpointed model for inference on new data."""
import argparse
import os
from pathlib import PurePath

import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import Tensor

from biobank_project.deep_mtl.models import bottleneck
from biobank_project.deep_mtl.training import utils


def get_argparser():
    parser = argparse.ArgumentParser(
        description='Infer on new metabolite data using checkpointed model.',
        conflict_handler='resolve')
    parser.add_argument(
        '-i', '--input', type=str, help='input csv with NBS metabolite values',
        required=True, metavar='INPUT_FILE', dest='input_file')
    parser.add_argument(
        '--id_column', type=str, help='column name for sample IDs',
        required=True, metavar='ID_COL')
    parser.add_argument(
        '-o', '--output', type=str, help='output directory for results',
        required=True, metavar='OUTPUT_DIR', dest='output_dir')
    parser.add_argument(
        '-s', '--state_dict', type=str,
        help='path of pytorch model state_dict')
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
    id_column = args.id_column
    output_dir = PurePath(args.output_dir)
    model_state_dict_file = args.state_dict

    # Read in data
    input_data = pd.read_csv(input_file, low_memory=False)
    input_data.set_index(id_column, inplace=True)

    # Enforce expected metabolite order for models
    # (Column order for metabolites during model training)
    metabolite_order = read_lines('./config/expected_metabolite_order.txt')
    data_X = input_data.loc[:, metabolite_order]
    data_X = pd.DataFrame(
        StandardScaler().fit_transform(data_X),
        columns=data_X.columns,
        index=data_X.index)

    # Enforce expected outcomes order
    outcomes = read_lines('./config/neonatal_covariates.txt')

    # Instantiate 1-unit bottleneck model
    utils.seed_torch(101)
    n_features = len(data_X.columns)
    n_tasks = len(outcomes)
    n_hidden = 100
    n_bottleneck = 1
    model = bottleneck.EnsembleNetwork(
        n_features=n_features, n_tasks=n_tasks,
        n_hidden=n_hidden, n_bottleneck=n_bottleneck
    )

    # Load state dict into model
    model.load_state_dict(torch.load(model_state_dict_file))

    # Set up model evaluation
    model.eval()
    model_output, bottleneck_output = model(
        torch.Tensor(data_X.values), return_bottleneck=True)

    # Create dataframes from results
    model_output_df = pd.DataFrame(
        torch.sigmoid(model_output).detach().numpy(),
        columns=outcomes,
        index=data_X.index)
    bottleneck_output_df = pd.DataFrame(
        bottleneck_output.detach().numpy(),
        columns=['bottleneck_output_' + str(i) for i in range(bottleneck_output.shape[1])],
        index=data_X.index)

    # Write out results
    os.makedirs(output_dir, exist_ok=True)
    write_results({
        'prediction_output': model_output_df,
        'bottleneck_output': bottleneck_output_df
    }, output_dir)


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    main(args)

