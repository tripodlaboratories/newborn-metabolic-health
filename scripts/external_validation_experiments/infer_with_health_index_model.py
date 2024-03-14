"""Use a checkpointed model for inference on Ontario data."""
import argparse
import json
import os
from pathlib import PurePath

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

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
        '-c', '--config', type=str, help='config file for model training')
    parser.add_argument(
        '-o', '--output', type=str, help='output directory for results',
        required=True, metavar='OUTPUT_DIR', dest='output_dir')
    parser.add_argument(
        '-s', '--state_dict', type=str,
        help='path of pytorch model state_dict')
    parser.add_argument(
        '--scaler_object', type=str, default=None,
        help='Pretrained scaler object to use for input data')
    return parser


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
    config_file = args.config
    output_dir = PurePath(args.output_dir)
    model_state_dict_file = args.state_dict
    scaler_object = args.scaler_object

    with open(config_file) as f:
        config = json.load(f)

    # Read in data
    id_column = config.get("id_column")
    input_data = pd.read_csv(input_file, low_memory=False)
    input_data.set_index(id_column, inplace=True)

    # Enforce expected metabolite order for models
    # (Column order for metabolites during model training)
    metabolite_order = config.get('metabolites')
    data_X = input_data.loc[:, metabolite_order]


    # Option for scaler or to instantiate new scaler
    if scaler_object is None:
        scaler = StandardScaler()
    else:
        scaler = joblib.load(scaler_object)

    scaled_data = scaler.fit_transform(data_X)
    data_X = pd.DataFrame(
        scaled_data,
        columns=data_X.columns,
        index=data_X.index)

    # The number of outcomes is used only in model instantiation
    outcomes = config.get('outcomes')

    # Instantiate 1-unit bottleneck model
    utils.seed_torch(101)
    n_features = len(data_X.columns)
    n_tasks = len(outcomes)
    n_hidden = 100
    n_bottleneck = 1
    model = bottleneck.EnsembleNetwork(
        n_features=n_features, n_tasks=n_tasks,
        n_hidden=n_hidden, n_bottleneck=n_bottleneck)

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
        columns=['health_index_score_' + str(i) for i in range(bottleneck_output.shape[1])],
        index=data_X.index)

    # Write out results
    os.makedirs(output_dir, exist_ok=True)
    write_results({
        'prediction_output': model_output_df,
        'health_index_output': bottleneck_output_df
    }, output_dir)


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    main(args)
