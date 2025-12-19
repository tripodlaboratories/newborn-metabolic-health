"""Tests for model training handlers."""
import pytest
from collections import namedtuple
import numpy as np
import pandas as pd
from torch import nn, Tensor
from torch.optim import Adam
from sklearn.datasets import (make_classification,
                              make_multilabel_classification,
                              make_regression)
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.model_selection import train_test_split
from unittest.mock import MagicMock, patch

from biobank_project.sklearn_interface import handlers as MOD

@pytest.fixture(name='RegressionDataMaker', scope='class')
def dataset_fixture():
    class RegressionDataMaker:
        def __init__(self,
        n_samples=500,
        n_features=30,
        n_targets=3,
        random_state=100):
            self.n_regression = n_targets
            self.n_features = n_features
            X, Y = make_regression(
                n_samples=n_samples, n_features=n_features,
                n_targets=self.n_regression,
                random_state=random_state)
            self.X = pd.DataFrame(
                X, columns=['feature_' + str(i) for i in range(n_features)])
            self.Y = pd.DataFrame(
                Y, columns=['reg_output' + str(i) for i in range(n_targets)])
    return RegressionDataMaker

class TestRegressionHandler:
    @pytest.fixture
    def dataset(self, RegressionDataMaker):
        return RegressionDataMaker()

    @pytest.fixture
    def model(self):
        return MultiTaskElasticNetCV()

    @pytest.fixture
    def training_runner(self, model):
        return MOD.RegressionTraining(model=model)

    def test_handler_sets_data(self, training_runner, dataset):
        colnames = dataset.Y.columns
        X_train, X_test, Y_train, Y_test = train_test_split(dataset.X, dataset.Y)
        training_runner.set_training_data(X_train, Y_train)
        training_runner.set_test_data(X_test, Y_test)
        assert hasattr(training_runner, 'X_train')
        assert hasattr(training_runner, 'Y_train')
        assert hasattr(training_runner, 'X_test')
        assert hasattr(training_runner, 'Y_test')

    def test_handler_training(self, training_runner, dataset):
        colnames = dataset.Y.columns
        X_train, X_test, Y_train, Y_test = train_test_split(dataset.X, dataset.Y)
        training_runner.set_training_data(X_train, Y_train)
        training_runner.set_test_data(X_test, Y_test)
        results = training_runner.train(colnames=colnames)
        assert 'preds' in results
        assert results['preds'].shape == Y_test.shape
