"""Repeated KFold cross validation for the sklearn interface"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


from biobank_project.deep_mtl.training.utils import (multilabel_to_single_array,
                                                     score_predictions,
                                                     score_regression)


class RepeatedKFold:
    """Class for running repeated KFold cross validation"""
    def __init__(
        self,
        n_folds: int,
        n_iter: int,
        data_X: pd.DataFrame,
        data_Y: pd.DataFrame,
        training_handler,
        X_valid: pd.DataFrame=None,
        Y_valid: pd.DataFrame=None,
        splitter: callable=None,
        output_type: str='regression'):
        """
        args:
            n_folds: number of folds to split data
            n_folds: number of iterations to perform repeated KFold training
            data_X: DataFrame containing features for training
            data_Y: DataFrame containing outcomes of interest
            training_handler: object which controls model training
            X_valid: External validation data for model (optional)
            Y_valid: External validation outcomes for model (optional),
            splitter: KFold splitting object
            output_type: 'classification' or 'regression'
        """
        self.n_folds = n_folds
        self.n_iter = n_iter
        self.data_X = data_X
        self.data_Y = data_Y
        self.training_handler = training_handler
        self.X_valid = X_valid
        self.Y_valid = Y_valid

        if splitter is None:
            self.splitter = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True,
                random_state=np.random.randint(0, 500))
        else:
            self.splitter = splitter

        ALLOWED_MODES = ['classification', 'regression']
        if output_type not in ALLOWED_MODES:
            raise ValueError(
                'Specification of KFold mode must be one of ' +
                str(ALLOWED_MODES))
        else:
            self.output_type = output_type

    def repeated_kfold(self,
        training_args: dict,
        resampler=None,
        splitting_cols: list=None):
        """args:
            training_args: dict of args for self.training_handler.train()
            n_iter: number of iterations of repeated KFold to run
            resampler: Resampler object (e.g., for imbalanced data)
            splitting_cols: columns to use in defining KFold splits
        """
        results_over_iters = {'scores': []}
        for i in range(self.n_iter):
            self.training_handler.reset_model()
            kfold_results = self.kfold(
                training_args=training_args, resampler=resampler,
                splitting_cols=splitting_cols)
            kfold_keys = kfold_results.keys()

            # Append keys from results to overall results
            if not all(k in results_over_iters.keys() for k in kfold_keys):
                extended_results_dict = {k: [] for k in kfold_keys}
                results_over_iters.update(extended_results_dict)
            for k in kfold_keys:
                if isinstance(kfold_results[k], pd.DataFrame):
                    kfold_results[k]['iter'] = i

            preds_df = kfold_results['preds']

            # Score predictions
            score_dfs = []
            if self.output_type == 'classification':
                class_tasks = self.data_Y.columns
                class_score = score_predictions(
                    preds=preds_df[class_tasks],
                    true_values=kfold_results['true_vals'][class_tasks],
                    columns_to_score=class_tasks)
                score_dfs.append(class_score)
            if self.output_type == 'regression':
                reg_tasks = self.data_Y.columns
                reg_score = score_regression(
                    preds=preds_df[reg_tasks],
                    true_values=kfold_results['true_vals'][reg_tasks],
                    columns_to_score=reg_tasks)
                score_dfs.append(reg_score)

            score = pd.concat(score_dfs)
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
    splitting_cols: list=None):
        """args:
            training_args: dict of args for self.training_handler.train()
            resampler: Resampler object (e.g., for imbalanced data)
            splitting_cols: columns to use in defining KFold splits
        """
        if splitting_cols is None and self.output_type == 'classification':
            encoded_y = multilabel_to_single_array(self.data_Y)
            kfold_args = [self.data_X, encoded_y]
        elif splitting_cols is None and self.output_type == 'regression':
            kfold_args = [self.data_X]
        else:
            encoded_y = multilabel_to_single_array(self.data_Y[splitting_cols])
            kfold_args = [self.data_X, encoded_y]

        results_over_folds = []
        true_values_over_folds = []
        for i, (train, test) in enumerate(self.splitter.split(*kfold_args)):
            # Reset model training and early stopping
            self.training_handler.reset_model()
            X_train = self.data_X.iloc[train, :]
            Y_train = self.data_Y.iloc[train, :]
            X_test = self.data_X.iloc[test, :]
            Y_test = self.data_Y.iloc[test, :]

            if resampler is not None:
                X_train, Y_train = resampler.resample(X_train, Y_train)

            # Data preparation
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                index=X_train.index, columns=X_train.columns)
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                index=X_test.index, columns=X_test.columns)
            self.training_handler.set_training_data(X_train_scaled, Y_train)
            self.training_handler.set_test_data(X_test_scaled, Y_test)
            if self.X_valid is not None and self.Y_valid is not None:
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
