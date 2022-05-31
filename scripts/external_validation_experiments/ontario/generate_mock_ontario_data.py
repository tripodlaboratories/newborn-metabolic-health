"""Generate mock data for testing - matches Ontario data specs."""
import argparse
from pathlib import PurePath
import numpy as np
import pandas as pd
from sklearn.datasets import make_multilabel_classification


def get_args():
    parser = argparse.ArgumentParser(
        description='Create mock Ontario data for testing',
        conflict_handler='resolve')
    parser.add_argument(
        '--data_headers', default=None,
        help='One row CSV file containing header names matching true data features')
    parser.add_argument(
        '--outcome_headers', default=None,
        help='One row CSV file contain header names that match the outcomes data')
    parser.add_argument(
        '--id_col', default='b_ikn',
        help='Column to set as index in the dataframes')
    parser.add_argument(
        '--n_samples', default=1000, type=int,
        help='Number of samples to generate for mock data.')
    parser.add_argument(
        '-o', '--output_dir',
        help='Output directory for mock data.')
    return parser.parse_args()


def main(args):
    data_header_file = args.data_headers
    outcome_header_file = args.outcome_headers
    id_col = args.id_col
    n_samples = args.n_samples
    output_dir = args.output_dir

    # Infer the number of columns from the header files
    data_headers = pd.read_csv(data_header_file).set_index(id_col)
    outcome_headers = pd.read_csv(outcome_header_file).set_index(id_col)

    # Infer the number of outcomes from the header files
    OUTCOME_NAMES = ['bpd', 'rop', 'ivh', 'nec']
    outcome_cols_only = outcome_headers[OUTCOME_NAMES]

    # Create dataset
    rs = np.random.RandomState(seed=100)
    mock_X, mock_Y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=len(data_headers.columns),
        n_labels=2,
        n_classes=len(outcome_cols_only.columns),
        random_state=rs)

    # Generate a random integer index?
    index_gen = np.random.default_rng(seed=100)
    index = index_gen.integers(
        low=10000, high=1000000, size=mock_X.shape[0])
    mock_features = pd.DataFrame(
        mock_X, columns=data_headers.columns,
        index=index)
    mock_features.index.name = id_col
    mock_outcomes = pd.DataFrame(
        mock_Y, columns=outcome_cols_only.columns,
        index=index)
    mock_outcomes.index.name = id_col

    # Add back extra columns
    for col in outcome_headers:
        if col not in mock_outcomes.columns:
            mock_outcomes[col] = 'FILLER_ICD10_CODE'

    # Write out simulated data
    output = PurePath(output_dir)
    mock_features.to_csv(
        str(output.joinpath('mock_ontario_metab.csv')))
    mock_outcomes.to_csv(
        str(output.joinpath('mock_ontario_outcomes.csv')))


if __name__ == '__main__':
    args = get_args()
    main(args)
