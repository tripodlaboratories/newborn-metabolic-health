"""Run pre-trained model on Mednax data"""
import argparse
from pathlib import PurePath

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch


def get_argparser():
    parser = argparse.ArgumentParser(
        description='Create a checkpointed model for the 1-unit bottleneck.',
        conflict_handler='resolve')
    parser.add_argument(
        '-i', '--input_data', type=str,
        default='./external_validation/mednax/processed/mednax_metabolites.csv',
        help='input data for model training.')
    parser.add_argument(
        '--model', type=str,
        default='./checkpoints/metabolic_health_index_1unit_bottleneck/checkpointed_model.pt',
        help='Saved pytorch model')
    parser.add_argument(
        '--scaler', type=str,
        default=None,
        help='Fitted scaler or data preprocessing pipeline')
    parser.add_argument(
        '-o', '--output', type=str,
        help='Output directory to store prediction results.')
    parser.add_argument(
        '--null_inf_cols', action='store_true', default=False,
        help='Handle null and inf cols in data to be used for prediction')
    return parser


def validate_data_format(
        reference_features,
        new_dataset_features,
        matching_df):
    pass


def main(args):
    # Read in Mednax data
    mednax_features = args.input_data
    mednax_metab = pd.read_csv(mednax_features, low_memory=False)
    # Drop extra ID columns, use one of the id columns as an index
    mednax_metab = mednax_metab.drop(
        columns='ID', errors='ignore').set_index('QuestionsRCode')

    # Load up pytorch model
    checkpointed_model = args.model
    model = torch.load(checkpointed_model)

    # TODO? Implement a series of data checks for the California metabolite data
    # order and the mednax data
    # validate_data_format()

    # Scale Mednax data using the preprocessing scaler from California
    # Biobank Data (we expect the data distributions to be considerably
    # different, such that a scaler fit on the California dataset would be
    # brittle to these changes)
    saved_scaler = args.scaler
    if saved_scaler is None:
        scaler = StandardScaler()
    else:
        scaler = joblib.load(saved_scaler)

    # Scaling cannot occur when the metabolites contain NA values or
    # infinite values.
    # TODO: Question: Should these be output to the logger? Probably helpful.
    handle_null_and_inf_cols = args.null_inf_cols
    if handle_null_and_inf_cols:
        cols_with_null = mednax_metab.isnull().any(axis=0)
        null_cols = cols_with_null[cols_with_null == True].index.tolist()
        mednax_metab = mednax_metab.fillna(0)

        samples_with_inf = np.isinf(mednax_metab).any(axis=1)
        n_samples_with_inf = samples_with_inf.sum()
        mednax_metab = mednax_metab[~samples_with_inf]

    if saved_scaler is None:
        scaled_metabolites = scaler.fit_transform(mednax_metab.values)
    else:
        scaled_metabolites = scaler.transform(mednax_metab.values)

    # Convert scaled Mednax input to tensors
    feature_tensor = torch.tensor(scaled_metabolites).float()

    # Run inference on the Mednax data, where the metabolic health index is the
    # second object in a tuple
    model_output = model(feature_tensor, return_bottleneck=True)
    health_index = model_output[1]

    # Save health index output as dataframe
    health_index_df = pd.DataFrame(
        health_index.detach().numpy(),
        index=mednax_metab.index, columns=['health_index'])

    output_dir = PurePath(args.output)
    health_index_df.to_csv(output_dir.joinpath(
        'health_index_output.csv'))


if __name__ == '__main__':
    args = get_argparser().parse_args()
    main(args)
