"""Utility functions for training."""
import os
import random
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset

from biobank_project.deep_mtl.data import structures


def seed_torch(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def df_to_datastructs(*dfs: pd.DataFrame, batch_size: int, shuffle: bool, index=None) -> tuple:
    if all(isinstance(df, pd.DataFrame) for df in dfs):
        df_tensors = tuple(torch.Tensor(df.values) for df in dfs)
    else:
        df_tensors = tuple(torch.Tensor(df) for df in dfs)
    dataset = structures.IndexedTensorDataset(*df_tensors, reference_index=index)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return (dataset, dataloader)


def multilabel_to_single_array(Y: pd.DataFrame) -> np.ndarray:
    """Convert multiple columns to single array listing unique values.
    Result can then be used as a replacement y for stratified kfold.
    """
    # Labelencoder can operate on list of strings to encode them into
    # integers corresponding to all unique strings
    return LabelEncoder().fit_transform([''.join(str(row)) for row in Y.values])


def score_predictions(
    preds: pd.DataFrame, true_values: pd.DataFrame, columns_to_score: list,
    pos_label=1) -> pd.DataFrame:
    score_dfs = []
    for col in columns_to_score:
        col_df = pd.DataFrame.from_dict(
            {'neg_preds': 1 - preds[col].values,
             'pos_preds': preds[col].values,
             'true_vals': true_values[col].values,
             'total_conditions': true_values[columns_to_score].sum(axis=1)
             })

        try:
            _, col_pval = mannwhitneyu(
                x=col_df[col_df['true_vals'] == pos_label]['pos_preds'],
                y=col_df[col_df['true_vals'] != pos_label]['pos_preds'],
                use_continuity=True, alternative='two-sided')
        except ValueError:
            warnings.warn(
                'Error in calculating P Value, returning NaN', UserWarning)
            col_pval = np.nan

        col_auroc = roc_auc_score(
            y_true=col_df['true_vals'],
            y_score=col_df['pos_preds'],
            average=None
        )
        col_aupr = average_precision_score(
            y_true=col_df['true_vals'],
            y_score=col_df['pos_preds'],
            average='weighted'
        )

        # Caculate auroc and aupr using strict negatives as 0
        # i.e., negative conditions without overlap from others
        strict_ctrl_df = col_df[
            -((col_df['true_vals'] == 0) & (col_df['total_conditions'] > 0))]
        col_auroc_strict = roc_auc_score(
            y_true=strict_ctrl_df['true_vals'],
            y_score=strict_ctrl_df['pos_preds'],
            average=None
        )
        col_aupr_strict = average_precision_score(
            y_true=strict_ctrl_df['true_vals'],
            y_score=strict_ctrl_df['pos_preds'],
            average='weighted'
        )

        score_dfs.append(pd.DataFrame.from_dict({
            'task': [col],
            'pval': [col_pval],
            'auroc': [col_auroc],
            'aupr': [col_aupr],
            'auroc_strict': [col_auroc_strict],
            'aupr_strict': [col_aupr_strict]
        }))
    return pd.concat(score_dfs)


def score_regression(
    preds: pd.DataFrame, true_values: pd.DataFrame, columns_to_score: list):
    score_dfs = []
    for col in columns_to_score:
        col_preds, col_true_vals = preds[col].values, true_values[col].values
        col_spearman, col_pval = spearmanr(col_preds, col_true_vals)
        col_mae = mean_absolute_error(col_true_vals, col_preds)
        col_mse = mean_squared_error(col_true_vals, col_preds)
        col_r2 = r2_score(col_true_vals, col_preds)
        score_dfs.append(pd.DataFrame.from_dict({
            'task': [col],
            'spearman_rho': [col_spearman],
            'pval': [col_pval],
            'mae': [col_mae],
            'mse': [col_mse],
            'r2': [col_r2]
            }))
    return pd.concat(score_dfs)


def get_pos_weight(data_column: pd.Series,
    pos_label: int=1,
    return_all_weights: bool=False) -> float:
    """Expects to operate on pandas columns.
    """
    n_classes = len(data_column.unique())
    class_counts = data_column.value_counts()
    n_samples = len(data_column)

    if return_all_weights is True:
        weights = [n_samples / (class_counts[label] + 1e-3)
            for label in class_counts.index.sort_values()]
        return tuple(weights)
    else:
        return n_samples / (class_counts[pos_label] + 1e-3)
