"""Script for calculating feature importances from a trained bottleneck model."""
from argparse import ArgumentParser

from captum.attr import FeaturePermutation, IntegratedGradients, DeepLiftShap
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from biobank_project.deep_mtl.models import bottleneck

default_config = {
    'n_hidden': 100,
    'n_bottleneck': 1
}

def get_args():
    parser = ArgumentParser('Feature Importance Specific Use-Case Script')
    parser.add_argument(
        '-i', '--input_dir', type=str,
        help='Results directory with: X_test.csv, Y_test.csv, model_state_dict.pt')
    parser.add_argument(
        '-o', '--output_dir',
        help='Output directory for interpretability CSV file(s)')
    return parser.parse_args()


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Load test data
    X_test = pd.read_csv(f"{input_dir}X_test.csv").set_index("row_id")
    Y_test = pd.read_csv(f"{input_dir}Y_test.csv").set_index("row_id")

    # Instantiate the model and load the state dict
    n_features = X_test.shape[1]
    n_tasks = Y_test.shape[1]
    n_hidden = 100
    n_bottleneck = 1
    model = bottleneck.EnsembleNetwork(
        n_features=n_features, n_tasks=n_tasks,
        n_hidden=n_hidden, n_bottleneck=n_bottleneck)
    model.load_state_dict(torch.load(f"{input_dir}model_state_dict.pt"))

    # Feature interpretability main loop
    avg_interp_results = []
    ind_interp_results = []
    input_tensor = torch.from_numpy(X_test.values).float()
    label_tensor = torch.from_numpy(Y_test.values).float()

    # Integrated Gradients
    ig = IntegratedGradients(model)
    ig_input_tensor = input_tensor.clone().detach()
    ig_input_tensor.requires_grad_()
    attr, delta = ig.attribute(ig_input_tensor, target=1, return_convergence_delta=True)
    attr = attr.detach().numpy()

    ig_df = pd.DataFrame.from_dict({
        'features': X_test.columns,
        'importance_score': np.mean(attr, axis=0)})
    ig_df['method'] = 'Integrated Gradients'
    avg_interp_results.append(ig_df)

    ind_ig_df = pd.DataFrame(attr, index=X_test.index, columns=X_test.columns)
    ind_ig_df['method'] = 'Integrated Gradients'
    ind_interp_results.append(ind_ig_df.reset_index())

    # Feature Permutation
    fp = FeaturePermutation(model)
    fp_attr = fp.attribute(input_tensor, target=1)
    fp_attr = fp_attr.detach().numpy()
    fp_df = pd.DataFrame.from_dict({
        'features': X_test.columns,
        'importance_score': np.mean(fp_attr, axis=0)})
    fp_df['method'] = 'Feature Permutation'
    avg_interp_results.append(fp_df)

    ind_fp_df = pd.DataFrame(fp_attr, index=X_test.index, columns=X_test.columns)
    ind_fp_df['method'] = 'Feature Permutation'
    ind_interp_results.append(ind_fp_df.reset_index())

    # DeepLiftSHAP
    dls = DeepLiftShap(model)
    splits = [700, 300]
    attr_tensor, baseline_tensor = torch.split(input_tensor, splits)
    attr_index = X_test.index[0:splits[0]]
    dls_attr, dls_delta = dls.attribute(
        attr_tensor, baseline_tensor, target=1,
        return_convergence_delta=True)
    dls_attr = dls_attr.detach().numpy()
    dls_df = pd.DataFrame.from_dict({
        'features': X_test.columns,
        'importance_score': np.mean(dls_attr, axis=0)})
    dls_df['method'] = 'DeepLiftSHAP'
    avg_interp_results.append(dls_df)

    ind_dls_df = pd.DataFrame(dls_attr, index=attr_index, columns=X_test.columns)
    ind_dls_df['method'] = 'DeepLiftSHAP'
    ind_interp_results.append(ind_dls_df.reset_index())

    # Write out results
    avg_interp_df = pd.concat(avg_interp_results)
    avg_interp_df.to_csv(f'{output_dir}average_scores.csv', index=False)

    ind_interp_df = pd.concat(ind_interp_results)
    ind_interp_df.to_csv(f'{output_dir}individual_scores.csv', index=False)

if __name__ == '__main__':
    args = get_args()
    main(args)