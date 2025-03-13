"""Add additional metadata features."""
from argparse import ArgumentParser
import numpy as np
import pandas as pd


def get_args():
    parser = ArgumentParser(
        description='Script for adding and recoding additional metadata features in addition to metabolites.',
        conflict_handler='resolve')
    parser.add_argument(
        '--input', type=str, default='data/processed/neonatal_conditions.csv',
        help='Input data that has newborn metabolites and outcomes.')
    parser.add_argument(
        '--metadata', type=str, default='data/processed/metadata.csv',
        help='Metadata CSV to draw additional features from')
    parser.add_argument('--id_col', type=str, default='row_id', help='Column name for the ID column')
    parser.add_argument('--output', help='Output file for augmented neonatal features and outcomes CSV')
    return parser.parse_args()


# Data-related encoding constants that may be useful
metadata_columns = [
    'mrace_catm', # Mother's race/ethnicity
    'mage_catm2', # Mother's age
    'frace_catm', # Father's race/ethnicity
    'sex3', # Infant's sex at birth
]
race_ethnicity_key = {
    1: 'Non-Hispanic White',
    2: 'Non-Hispanic Black',
    3: 'Asian',
    4: 'Pacific Islander',
    5: 'Hispanic',
    6: 'American Indian/Alaskan Native',
    7: 'Other',
    99: np.nan
}
age_categorical_key = {
    1: '11-19',
    2: '20-24',
    3: '25-29',
    4: '30-34',
    5: '35-39',
    6: '40-44',
    7: '45-70',
    99: np.nan
}
infant_sex_categorical_key = {
    1: 'Male',
    2: 'Female'
}


def main(args):
    input_data = args.input
    id_col = args.id_col
    metadata_file = args.metadata
    output_file = args.output

    input_data = pd.read_csv(input_data).set_index(id_col)
    meta = pd.read_csv(metadata_file, low_memory=False).set_index(id_col)

    # Collect specific metadata features and recode for one-hot
    # Currently this adds infant sex to the dataframe
    # First perform any necessary recoding (e.g., one-hot) for downstream analysis
    encoded_meta = meta[['sex3']].copy()
    infant_sex_cat_values = {v: k for k, v in infant_sex_categorical_key.items()}
    male_encoding = infant_sex_cat_values['Male']
    encoded_meta['male_infant_sex'] = [1 if col == male_encoding else 0 for col in encoded_meta['sex3']]
    encoded_meta.drop(columns=['sex3'], inplace=True)

    input_meta = pd.merge(
        input_data,
        encoded_meta,
        left_index=True,
        right_index=True
    )
    input_meta.to_csv(output_file, index=True)


if __name__ == '__main__':
    args = get_args()
    main(args)
