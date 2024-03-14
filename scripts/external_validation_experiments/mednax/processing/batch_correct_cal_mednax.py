"""Batch correction pipeline for California and Mednax."""
import argparse
from pathlib import PurePath

from combat.pycombat import pycombat
import pandas as pd


def get_argparser():
    parser = argparse.ArgumentParser(
        description='Batch correction between California and Mednax.',
        conflict_handler='resolve')
    parser.add_argument(
        '--california_dir',
        default='./data/processed/',
        help='Input experiment directory, should contain bottleneck.csv, valid_bottleneck.csv, and true_vals.csv')
    parser.add_argument(
        '--mednax_dataset',
        default='./external_validation/mednax/processed/mednax_metabolites_cal_names.csv',
        help='Metabolites file corresponding to validation data, metabolites must match.')
    parser.add_argument(
        '--metabolite_order_file',
        default='./config/expected_metabolite_order.txt',
        help='File containing expected metabolite order.')
    parser.add_argument(
        '--gestational_age_range',
        default='./config/gestational_age_ranges.txt',
        help='File containing gestational age range list for California data')
    parser.add_argument(
        '-o', '--output_dir',
        help='Output directory for batch corrected results')
    return parser


def main(args):
    cal_dir = PurePath(args.california_dir)
    mednax_dataset = PurePath(args.mednax_dataset)
    metabolite_order_file = args.metabolite_order_file
    ga_range_file = args.gestational_age_range
    output_dir = PurePath(args.output_dir)

    # Requires input, the Mednax data with the California Biobank names.
    # Read in the Mednax data and get the same dimensions as the California dataset
    mednax = pd.read_csv(
        mednax_dataset)
    mednax.drop(columns=['ID'], inplace=True)
    mednax.set_index('QuestionsRCode', inplace=True)

    # Prepare data to perform dimensionality reduction using the California subset.
    with open(metabolite_order_file) as f:
        cal_metabolite_order = [l.strip() for l in f.readlines()]

    cal_biobank_data = pd.read_csv(
        str(cal_dir.joinpath('neonatal_conditions.csv')), low_memory=False).set_index('row_id')
    meta = pd.read_csv(str(cal_dir.joinpath('metadata.csv')), low_memory=False).set_index('row_id')

    with open(ga_range_file) as f:
        preterm_ranges = [l.strip() for l in f.readlines()]
    preterm_meta = meta[meta.gacat.isin(preterm_ranges)]
    cal_preterm_metab = cal_biobank_data.loc[preterm_meta.index]

    # Then align the datasets
    cal_preterm_metab = cal_preterm_metab[cal_metabolite_order]
    cal_preterm_metab['dataset'] = 0
    cal_preterm_metab.dropna(
        axis=1, inplace=True)
    mednax_metab = mednax[cal_metabolite_order]
    mednax_metab.dropna(axis=1, inplace=True)
    mednax_metab['dataset'] = 1

    # Batch-correct data, keeping metabolites in common.
    joined_metab = pd.concat(
        [cal_preterm_metab, mednax_metab],
        join='inner', axis=0)
    batch_indicator = joined_metab['dataset'].to_numpy()
    joined_metab.drop(columns=['dataset'], inplace=True)
    metab_corrected = pycombat(
        joined_metab.transpose(),
        list(batch_indicator)).transpose()

    # Write out the dataset
    cal_corrected = metab_corrected.loc[batch_indicator == 0]
    cal_corrected.index.name = 'row_id'
    mednax_corrected = metab_corrected.loc[batch_indicator == 1]
    mednax_corrected.index.name = 'QuestionsRCode'

    cal_corrected.to_csv(output_dir.joinpath(
        'cal_metabolites_corrected.csv'))
    mednax_corrected.to_csv(output_dir.joinpath(
        'mednax_metabolites_corrected.csv'))

    # Create a version that's joined with outcomes
    # This is the expected format for some of the model training scripts
    outcome_cols = ['nec_any', 'rop_any', 'bpd_any', 'ivh_any']
    cal_corrected_with_outcomes = pd.merge(
        cal_corrected, cal_biobank_data[outcome_cols],
        left_index=True, right_index=True)
    cal_corrected_with_outcomes.to_csv(output_dir.joinpath(
        'cal_metabolites_outcomes_corrected.csv'))


if __name__ == '__main__':
    args = get_argparser().parse_args()
    main(args)
