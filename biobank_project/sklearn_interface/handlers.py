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


class RegressionModelTraining:
    """Class for running model training.
    """
    def __init__(self, model):
        """
        args:
            model: scikit-learn model
        """
        self.model = model
        self.init_model = clone(model)

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
        preds = self.model.predict(input_data)
        return pd.DataFrame(preds, columns=colnames, index=index)

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
