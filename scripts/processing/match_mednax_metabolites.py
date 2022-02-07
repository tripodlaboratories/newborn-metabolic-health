"""Mednax metabolite matching to California NBS."""
import argparse
from pathlib import PurePath

import numpy as np
import pandas as pd

from biobank_project.io import read_lines


def main():
    # Read in matching data
    mednax_fd = './external_validation/mednax'
    mednax_dir = PurePath(mednax_fd)
    metabmatch_dir = mednax_dir.joinpath('metabolite_matching')
    matching_df = pd.read_csv(
        metabmatch_dir.joinpath('calbiobank_mednax_metab_matching.csv'))

    # Read in California State Biobank Data
    cal_biobank_dir = PurePath('./data/processed/neonatal_conditions.csv')
    cal_biobank_metab = read_lines('./config/expected_metabolite_order.txt')

    # Read in Mednax data
    mednax = pd.read_csv(
        mednax_dir.joinpath('excel_processed_csvs', 'laboratory_data.csv'))
    mednax_demo_outcomes = pd.read_csv(
        mednax_dir.joinpath(
            'excel_processed_csvs', 'demographics_outcomes.csv'))

    # In most of the metabolic health index analyses, sparse columns are
    # dropped, the 'expected_metabolite_order.txt' file contains metabolites
    # after sparse features are dropped
    matching_df = matching_df[matching_df.cal_biobank.isin(cal_biobank_metab)]

    # Begin metabolite matching process
    # Derive missing ratios
    # Slice matching dataframe to find metabolite ratios that need derivation
    ratios_df = matching_df[matching_df['ratio_needs_derivation'] == 'Yes']
    ratios_to_derive = ratios_df['derived_ratio_name']
    for r in ratios_to_derive:
        derivation_data = ratios_df[ratios_df['derived_ratio_name'] == r]
        rn = derivation_data['ratio_numerator'].item()
        rd = derivation_data['ratio_denominator'].item()
        mednax[r] = mednax[rn] / mednax[rd]

    # Derive dummy NA columns for metabolites that are present in the
    # California NBS but absent in Mednax
    absent_metab_ix = np.where(
        matching_df['mednax'].isnull() &
        matching_df['ratio_needs_derivation'].isnull())
    absent_metab_df = matching_df.iloc[absent_metab_ix]
    absent_metabs = absent_metab_df['cal_biobank']

    for absent in absent_metabs:
        mednax[absent] = np.nan

    # Re-order to match the California biobank metabolite order
    matching_df['mednax_completed_metab'] = matching_df['mednax']
    matching_df.mednax_completed_metab.fillna(
        matching_df.derived_ratio_name, inplace=True)
    matching_df.mednax_completed_metab.fillna(
        matching_df.cal_biobank, inplace=True)
    mednax_metabolites = mednax[
       ['ID', 'QuestionsRCode'] +
        matching_df.mednax_completed_metab.tolist()]

    # Match outcomes between Mednax and the California Biobank
    original_outcomes_cols = [
        'IVH', 'ROP', 'Acquired.bowel.disease', 'X42daystatus']
    mednax_outcomes = mednax_demo_outcomes.set_index('QuestionsRCode')[
        original_outcomes_cols].copy()

    # Derive chronic lung disease, nec_any, ivh_any, rop_any columns and
    # match with the California Biobank data
    ivh_pos_indicators = [
        'Intraventricular', 'Intraventricular with dilation',
        'Subependymal', 'Intraparenchymal']
    mednax_outcomes['ivh_any'] = np.where(
        mednax_outcomes.IVH.isin(ivh_pos_indicators), 1, 0)

    rop_pos_indicators = ['ROP 1', 'ROP 2', 'ROP 3', 'ROP surgery']
    mednax_outcomes['rop_any'] = np.where(
        mednax_outcomes.ROP.isin(rop_pos_indicators), 1, 0)

    nec_pos_indicators = ['NEC medical', 'NEC surgical']
    mednax_outcomes['nec_any'] = np.where(
        mednax_outcomes['Acquired.bowel.disease'].isin(nec_pos_indicators), 1, 0)

    # Not strictly BPD, but we can have some indicator of chronic lung disease
    mednax_outcomes['bpd_any'] = np.where(
        mednax_outcomes['X42daystatus'] == 'Alive on oxygen', 1, 0)

    new_outcome_cols = ['nec_any', 'rop_any', 'bpd_any', 'ivh_any']
    mednax_outcomes = mednax_outcomes[new_outcome_cols]

    id_cols = ['ID', 'QuestionsRCode']
    id_for_merge = 'QuestionsRCode'
    feat_outcome_columns = set(
        mednax_metabolites.columns.tolist() + mednax_outcomes.columns.tolist())
    demo_columns = (set(mednax_demo_outcomes.columns)
                    .difference(feat_outcome_columns))
    demo_columns.update(set(mednax.columns).difference(feat_outcome_columns))
    merge_cols_to_use = mednax_demo_outcomes.columns.difference(mednax.columns)
    merge_cols_to_use = [id_for_merge] + merge_cols_to_use.tolist()
    all_merged = pd.merge(
        mednax.set_index(id_for_merge),
        mednax_demo_outcomes[merge_cols_to_use].set_index(id_for_merge),
        left_index=True, right_index=True, how='inner')
    mednax_demo = all_merged.reset_index()[id_cols + list(demo_columns)]

    # Select the demographics and lab values for day 1.
    # Then merge into outcomes and split
    mednax_demo_day1 = mednax_demo[mednax_demo['ID'].str.endswith(' 1')]
    mednax_outcomes_demo = pd.merge(
        mednax_outcomes,
        mednax_demo_day1.set_index(id_for_merge),
        left_index=True, right_index=True)

    # Merge and split dataframes into features + outcomes
    mednax_metab_day1 = mednax_metabolites[
        mednax_metabolites['ID'].str.endswith(' 1')].drop(columns='ID')
    merged = pd.merge(
        mednax_metab_day1.set_index('QuestionsRCode'),
        mednax_outcomes_demo,
        left_index=True, right_index=True).reset_index()

    features_from_merge = merged[mednax_metabolites.columns.tolist()]
    outcomes_from_merge = merged[id_cols + new_outcome_cols]
    demo_from_merge = merged[id_cols + list(demo_columns)]

    # Create a version of the Mednax metabolite data with California Biobank Labels
    features_cal_names = features_from_merge.copy()
    features_cal_names.rename(
        columns={k: v for k, v in zip(
            matching_df.mednax.fillna(matching_df.derived_ratio_name),
            matching_df.cal_biobank)
        }, inplace=True)

    # Write out processed data
    features_from_merge.to_csv(
        mednax_dir.joinpath('processed', 'mednax_metabolites.csv'),
        index=False)
    features_cal_names.to_csv(
        mednax_dir.joinpath('processed', 'mednax_metabolites_cal_names.csv'),
        index=False)
    outcomes_from_merge.to_csv(
        mednax_dir.joinpath('processed', 'mednax_outcomes.csv'),
        index=False)
    demo_from_merge.to_csv(
        mednax_dir.joinpath('processed', 'mednax_demo_meta_other.csv'),
        index=False)

if __name__ == '__main__':
    main()

