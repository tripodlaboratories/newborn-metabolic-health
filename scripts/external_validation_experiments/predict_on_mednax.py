"""Run pre-trained model on Mednax data"""
import argparse
from pathlib import PurePath

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch


def get_argparser():
    pass


def validate_data_format(
        reference_features,
        new_dataset_features,
        matching_df):
    pass


def main():
    # Read in Mednax data
    mednax_dir = PurePath('./external_validation/mednax/processed/')
    mednax_features = mednax_dir.joinpath('mednax_metabolites.csv')
    mednax_metab = pd.read_csv(mednax_features, low_memory=False)

    # Drop extra ID columns, use one of the id columns as an index
    mednax_metab = mednax_metab.drop(columns='ID').set_index('QuestionsRCode')

    # Load up pytorch model
    model_dir = PurePath(
        './checkpoints/metabolic_health_index_1unit_bottleneck/')
    model_file = model_dir.joinpath('checkpointed_model.pt')
    model = torch.load(str(model_file))

    # TODO? Implement a series of data checks for the California metabolite data
    # order and the mednax data
    # validate_data_format()

    # Scale Mednax data using the preprocessing scaler from California
    # Biobank Data (we expect the data distributions to be considerably
    # different, such that a scaler fit on the California dataset would be
    # brittle to these changes)
    scaler = StandardScaler()

    # Scaling cannot occur when the metabolites contain NA values or
    # infinite values.
    # TODO: Question: Should these be output to the logger? Probably helpful.
    cols_with_null = mednax_metab.isnull().any(axis=0)
    null_cols = cols_with_null[cols_with_null == True].index.tolist()
    mednax_metab = mednax_metab.fillna(0)

    samples_with_inf = np.isinf(mednax_metab).any(axis=1)
    n_samples_with_inf = samples_with_inf.sum()
    mednax_metab = mednax_metab[~samples_with_inf]
    scaled_metabolites = scaler.fit_transform(mednax_metab.values)

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

    output_dir = PurePath('./results/external_validation/mednax/')
    health_index_df.to_csv(output_dir.joinpath(
        'health_index_output.csv'))

if __name__ == '__main__':
    main()

