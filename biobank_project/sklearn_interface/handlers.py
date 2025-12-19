"""scikit-learn Model Handlers
"""
import copy
import os
import random
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim, as_tensor
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class ModelTraining:
    """Class for running model training.
    """
    def __init__(self, model, model_type='classificatin', pos_label=1):
        """
        args:
            model: scikit-learn model
            pos_label: expected encoding of the positive label in y
                (usually 1 in {0, 1} encoded data),
            model_type
        """
        self.model = model
        self.init_model = clone(model)
        self.pos_label = pos_label
        self.model_type = model_type

    def reset_model(self):
        self.model = clone(self.init_model)

    def set_training_data(self, X_train: pd.DataFrame, Y_train: pd.DataFrame):
        assert sorted(X_train.index) == sorted(Y_train.index)
        self.X_train, self.Y_train = X_train, Y_train

    def set_test_data(self, X_test: pd.DataFrame, Y_test: pd.DataFrame):
        # Shuffle is set to False so that test set IDs can be matched
        assert sorted(X_test.index) == sorted(Y_test.index)
        self.X_test, self.Y_test = X_test, Y_test

    def set_validation_data(self, X_valid: pd.DataFrame, Y_valid: pd.DataFrame):
        assert sorted(X_valid.index) == sorted(Y_valid.index)
        self.X_valid, self.Y_valid = X_valid, Y_valid
        self.validation = True

    def predict(self, input_data, colnames, index):
        raise NotImplementedError('Implement prediction logic in sub-class')

    def train(self,
        colnames: list=None,
        output_training_preds: bool=False):
        """Run training over epochs for a Torch module given dataloaders.
        args:
            colnames: column names for the predictions
            output_train_preds: whether or not training predictions should be in output

        returns: dict with the following:
            'preds': dataframe with test set predictions over epochs

            optional:
            'train_preds': dataframe with training set predictions over epochs
        """
        if self.X_train is None or self.X_test is None:
            raise AttributeError('Data structures have not been set.')

        # Train
        # Handle case of single prediction, use ravel() to avoid
        # DataConversionWarning
        if len(self.Y_train.columns) == 1:
            self.model.fit(self.X_train, self.Y_train.values.ravel())
        else:
            self.model.fit(self.X_train, self.Y_train)

        if output_training_preds is True:
            train_preds = self.predict(
                self.X_train, colnames=colnames, index=self.X_train.index)

        # Collect test set and validation set results
        test_preds = self.predict(
            self.X_test, colnames=colnames, index=self.X_test.index)

        if hasattr(self, 'X_valid'):
            valid_preds = self.predict(
                self.X_valid, colnames=colnames, index=self.X_valid.index)

        training_output = {'preds': test_preds}
        if output_training_preds is True:
            training_output['train_preds'] = train_preds
        if hasattr(self, 'X_valid'):
            training_output['valid_preds'] = valid_preds
        return training_output


class ClassificationTraining(ModelTraining):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, input_data, colnames, index):
        preds = self.model.predict_proba(input_data)

        # Extract prediction probabilities corresponding to the positive label
        # MultiOutputClassifier helper returns predictions as a list, one
        # element per class
        if isinstance(preds, list):
            pred_probs = []
            for p in preds:
                pred_probs.append(p[:, self.pos_label])
            pred_probs = np.column_stack(pred_probs)
            return pd.DataFrame(pred_probs, columns=colnames, index=index)
        else:
            # Single array of prediction probabilities expected
            pred_probs = preds[:, self.pos_label]
            return pd.DataFrame(pred_probs, columns=colnames, index=index)


class RegressionTraining(ModelTraining):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, input_data, colnames, index):
        preds = self.model.predict(input_data)

        if isinstance(preds, list):
            all_preds = []
            for p in preds:
                all_preds.append(p[:, ])
            all_preds = np.column_stack(all_preds)
            return pd.DataFrame(all_preds, columns=colnames, index=index)
        else:
            return pd.DataFrame(preds, columns=colnames, index=index)
