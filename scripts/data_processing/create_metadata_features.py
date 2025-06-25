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
    'gacat', # Gestational Age Categorical Range
    'bwtcat', # Birthweight Categorical Range
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


def range_var_numerizer(df, cols_to_numerize, col_append='_numeric', split_char='_', dtype=np.float64):
    """Convert GA and Birthweight to the numerical value in the middle of the range.
    
    Args:
        df: DataFrame to modify
        cols_to_numerize: List of column names or single column name
        col_append: Suffix for new column names
        split_char: Character to split ranges on
        dtype: Target data type
    """
    if isinstance(cols_to_numerize, str):
        cols_to_numerize = [cols_to_numerize]
    
    for col in cols_to_numerize:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        new_col_name = f'{col}{col_append}'
        df[new_col_name] = df[col].apply(
            lambda x: (float(x.split(split_char)[0]) + float(x.split(split_char)[1])) / 2 
            if isinstance(x, str) and split_char in x else x
        ).astype(dtype)
    
    return df


def main(args):
    input_data = args.input
    id_col = args.id_col
    metadata_file = args.metadata
    output_file = args.output

    input_data = pd.read_csv(input_data).set_index(id_col)
    meta = pd.read_csv(metadata_file, low_memory=False).set_index(id_col)

    # Collect specific metadata features and recode for one-hot
    # Currently this adds infant sex, gestational age, and birthweight to the dataframe
    # First perform any necessary recoding (e.g., one-hot) for downstream analysis
    encoded_meta = meta[['sex3', 'gacat', 'bwtcat']].copy()
    infant_sex_cat_values = {v: k for k, v in infant_sex_categorical_key.items()}
    male_encoding = infant_sex_cat_values['Male']
    encoded_meta['male_infant_sex'] = [1 if col == male_encoding else 0 for col in encoded_meta['sex3']]

    # Then convert the categorical range columns of birthweight and gestational age to numerical mid-range values
    # Transform: gacat -> gacat_numeric, bwtcat -> bwtcat_numeric
    encoded_meta = range_var_numerizer(encoded_meta, cols_to_numerize=['gacat', 'bwtcat'], col_append='_numeric')

    # Drop the original column names that we recoded
    encoded_meta.drop(columns=['sex3', 'gacat', 'bwtcat'], inplace=True)

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
