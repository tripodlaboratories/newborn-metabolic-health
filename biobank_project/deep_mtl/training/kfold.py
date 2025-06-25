"""Repeated KFold training"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


from biobank_project.deep_mtl.training.handlers import ModelTraining
from biobank_project.deep_mtl.training.utils import (multilabel_to_single_array,
                                                     score_predictions,
                                                     score_regression)


def get_lowest_loss_preds(preds: pd.DataFrame, losses: pd.DataFrame):
    """Filter predictions only to keep predictions from the epoch with the lowest loss.
    This epoch may be different across different folds of repeated KFold.
    """
    # .transform('min') creates a new series that's the same length as the original
    # where each value is replaced by the group's minimum
    # This allows a comparison to losses['test_loss'] to create a boolean mask for indexing.
    lowest_loss_indices = (losses
        .groupby('fold')['test_loss']
        .transform('min') == losses['test_loss'])

    lowest_losses = losses.loc[lowest_loss_indices, :]
    index_name = preds.index.name
    preds = (preds
        .reset_index()
        .merge(lowest_losses, on=['epoch', 'fold', 'iter'])
        .rename(columns={'index': index_name})
        .set_index(index_name))
    preds = preds.drop(
        columns=['train_loss', 'test_loss', 'valid_loss'],
        errors='ignore')

    # To resolve the case of tied losses, take the lowest epoch in each fold
    highest_epoch_indices = (preds
        .groupby('fold')['epoch']
        .transform('min') == preds['epoch'])

    if highest_epoch_indices.tolist().count(True) < preds.shape[0]:
        # There are folds where lowest loss is tied
        # more than one prediction per sample
        # multiple predictions per sample -- may break downstream scoring
        preds = preds.loc[highest_epoch_indices, :]

    return preds


class RepeatedKFold:
    """Class for running repeated KFold cross validation"""
    def __init__(
        self,
        n_iter: int, n_folds: int,
        data_X: pd.DataFrame, data_Y: pd.DataFrame,
        training_handler: ModelTraining,
        X_valid: pd.DataFrame=None,
        Y_valid: pd.DataFrame=None):
        """
        args:
            n_iter: number of iterations of repeated kfold to run
            n_folds: number of folds to split data
            data_X: DataFrame containing features for training
            data_Y: DataFrame containing outcomes of interest
            training_handler: object which controls model training
            X_valid: External validation data for model (optional)
            Y_valid: External validation outcomes for model (optional)
        """
        self.n_iter = n_iter
        self.n_folds = n_folds
        self.data_X = data_X
        self.data_Y = data_Y
        self.training_handler = training_handler
        self.X_valid = X_valid
        self.Y_valid = Y_valid

    def repeated_kfold(self,
    training_args: dict,
    resampler=None,
    mixed_output: bool=None,
    class_tasks: list=[],
    reg_tasks: list=[],
    splitting_cols: list=None,
    columns_to_score: list=None,
    categorical_features: list = None):
        """args:
            training_args: dict of args for self.training_handler.train()
            resampler: Resampler object (e.g., for imbalanced data)
            mixed_output: whether or not data_Y contains mixed regression and classification
                if True, specify classification and regression tasks
            splitting_cols: columns to use in defining KFold splits
            columns_to_score: columns in Y used to evaluate predictions
            categorical_features: columns in data_X that should NOT be scaled.
        """
        if columns_to_score is None:
            columns_to_score = self.data_Y.columns

        if len(class_tasks) == 0:
            class_tasks = None
        if len(reg_tasks) == 0:
            reg_tasks = None

        results_over_iters = {'scores': []}
        for i in range(self.n_iter):
            self.training_handler.reset_model()
            kfold_results = self.kfold(
                training_args=training_args, resampler=resampler,
                mixed_output=mixed_output, splitting_cols=splitting_cols,
                categorical_features=categorical_features)
            kfold_keys = kfold_results.keys()

            # Append keys from results to overall results
            if not all(k in results_over_iters.keys() for k in kfold_keys):
                extended_results_dict = {k: [] for k in kfold_keys}
                results_over_iters.update(extended_results_dict)
            for k in kfold_keys:
                if isinstance(kfold_results[k], pd.DataFrame):
                    kfold_results[k]['iter'] = i

            # Use preds from the epoch with the lowest loss
            for k in kfold_keys:
                if k in ['preds', 'train_preds', 'valid_preds', 'bottleneck', 'valid_bottleneck']:
                    kfold_results[k] = get_lowest_loss_preds(
                    kfold_results[k], kfold_results['losses'])

            preds_df = kfold_results['preds']

            # Score predictions
            if mixed_output is True:
                score_dfs = []
                if class_tasks is not None:
                    class_score = score_predictions(
                        preds=preds_df[class_tasks],
                        true_values=kfold_results['true_vals'][class_tasks],
                        columns_to_score=class_tasks)
                    score_dfs.append(class_score)
                if reg_tasks is not None:
                    reg_score = score_regression(
                        preds=preds_df[reg_tasks],
                        true_values=kfold_results['true_vals'][reg_tasks],
                        columns_to_score=reg_tasks)
                    score_dfs.append(reg_score)
                score = pd.concat(score_dfs)
                score['iter'] = i
            else:
                score = score_predictions(
                    preds=preds_df,
                    true_values=kfold_results['true_vals'],
                    columns_to_score=columns_to_score)
                score['iter'] = i

            for k in kfold_keys:
                results_over_iters[k].append(kfold_results[k])
            results_over_iters['scores'].append(score)
        return {
            k: pd.concat(results_over_iters[k])
            for k in results_over_iters.keys()
        }

    def kfold(self,
    training_args: dict,
    resampler=None,
    mixed_output: bool=None,
    splitting_cols: list=None,
    categorical_features: list=None):
        """args:
            training_args: dict of args for self.training_handler.train()
            resampler: Resampler object (e.g., for imbalanced data)
            mixed_output: whether or not data_Y contains mixed regression and classification
            splitting_cols: columns to use in defining KFold splits
        """
        k_fold = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True,
            random_state=np.random.randint(0, 500))
        if splitting_cols is None:
            encoded_y = multilabel_to_single_array(self.data_Y)
            split_args = [self.data_X, encoded_y]
        elif splitting_cols == 'regression_only':
            k_fold = KFold(
                n_splits=self.n_folds, shuffle=True,
                random_state=np.random.randint(0, 500))
            split_args = [self.data_X]
        else:
            encoded_y = multilabel_to_single_array(self.data_Y[splitting_cols])
            split_args = [self.data_X, encoded_y]

        results_over_folds = []
        true_values_over_folds = []
        for i, (train, test) in enumerate(k_fold.split(*split_args)):
            # Reset model training and early stopping
            self.training_handler.reset_model()
            if training_args['early_stopping_handler'] is not None:
                training_args['early_stopping_handler'].reset()

            X_train = self.data_X.iloc[train, :]
            Y_train = self.data_Y.iloc[train, :]
            X_test = self.data_X.iloc[test, :]
            Y_test = self.data_Y.iloc[test, :]

            if resampler is not None:
                X_train, Y_train = resampler.resample(X_train, Y_train)

            # Data preparation
            scaler = StandardScaler()
            if categorical_features is not None:
                features_to_scale = X_train.columns.difference(categorical_features)
                X_train_scaled = X_train.copy()
                X_train_scaled[features_to_scale] = scaler.fit_transform(
                    X_train_scaled[features_to_scale])
                X_test_scaled = X_test.copy()
                X_test_scaled[features_to_scale] = scaler.transform(
                    X_test_scaled[features_to_scale])
            else:
                X_train_scaled = pd.DataFrame(
                    scaler.fit_transform(X_train),
                    index=X_train.index, columns=X_train.columns)
                X_test_scaled = pd.DataFrame(
                    scaler.transform(X_test),
                    index=X_test.index, columns=X_test.columns)

            self.training_handler.set_training_data(X_train_scaled, Y_train)
            self.training_handler.set_test_data(X_test_scaled, Y_test)
            if self.X_valid is not None and self.Y_valid is not None:
                if categorical_features is not None:
                    X_valid_scaled = self.X_valid.copy()
                    X_valid_scaled[features_to_scale] = scaler.transform(
                        X_valid_scaled[features_to_scale])
                else:
                    X_valid_scaled = pd.DataFrame(
                        scaler.transform(self.X_valid),
                        index=self.X_valid.index, columns=self.X_valid.columns)

                self.training_handler.set_validation_data(X_valid_scaled, self.Y_valid)

            fold_results = self.training_handler.train(**training_args)

            # Append fold number to results dataframes
            for k in fold_results.keys():
                if isinstance(fold_results[k], pd.DataFrame):
                    fold_results[k]['fold'] = i
            results_over_folds.append(fold_results)

            # Also keep true values
            true_values = Y_test.copy()
            true_values['fold'] = i
            true_values_over_folds.append(true_values)

        # Combine results
        # Extract corresponding results keys and create new dict
        results_keys = {k for d in results_over_folds for k in d.keys()}
        collected_results = dict.fromkeys(results_keys)
        for k in collected_results.keys():
            collected_results[k] = pd.concat(
                result[k] for result in results_over_folds)
        collected_results['true_vals'] = pd.concat(true_values_over_folds)
        return collected_results
