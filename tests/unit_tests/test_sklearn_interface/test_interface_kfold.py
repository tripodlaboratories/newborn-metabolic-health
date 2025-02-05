"""Tests for KFold Cross Validation."""
import pytest
from collections import namedtuple
import numpy as np
import pandas as pd
from sklearn.datasets import (make_classification,
                              make_multilabel_classification,
                              make_regression)
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.model_selection import KFold
from unittest.mock import MagicMock, patch


from biobank_project.sklearn_interface import kfold as MOD
from biobank_project.sklearn_interface import handlers


@pytest.fixture(name='RegressionDataMaker', scope='class')
def dataset_fixture():
    class RegressionDataMaker:
        def __init__(self,
        n_samples=250,
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


class TestKFold:
    @pytest.fixture
    def dataset(self, RegressionDataMaker):
        return RegressionDataMaker()

    @pytest.fixture
    def model(self):
        return MultiTaskElasticNetCV()

    @pytest.fixture
    def train_runner(self, model):
        return handlers.RegressionTraining(model)

    @pytest.fixture
    def kfold(self, dataset, train_runner):
        splitter = KFold(n_splits=3, shuffle=True, random_state=101)
        return MOD.RepeatedKFold(
            n_folds=3, n_iter=2, data_X=dataset.X, data_Y=dataset.Y,
            training_handler=train_runner, splitter=splitter,
            output_type='regression')

    @pytest.fixture
    def train_args(self, dataset):
        return {
            'colnames': dataset.Y.columns
        }

    def test_kfold_returns_matched_results_shape_to_data(self, kfold, train_args):
        results = kfold.kfold(train_args)
        preds, true_vals = results['preds'], results['true_vals']
        assert preds.shape == true_vals.shape
        assert len(preds.index.unique()) == len(true_vals.index.unique())
        assert preds.shape[0] == kfold.data_Y.shape[0]
        assert len(preds['fold'].unique()) == kfold.n_folds
        assert len(true_vals['fold'].unique()) == kfold.n_folds

    def test_repeated_kfold_has_values_for_each_iter(self, kfold, train_args):
        results = kfold.repeated_kfold(train_args)
        for k in results.keys():
            assert len(results[k]['iter'].unique()) == kfold.n_iter