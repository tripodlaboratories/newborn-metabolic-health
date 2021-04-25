"""Tests for the confounder network training."""
import pytest
from collections import namedtuple
import numpy as np
import pandas as pd
from torch import nn, Tensor
from torch.optim import Adam
from sklearn.datasets import (make_classification,
                              make_multilabel_classification,
                              make_regression)
from sklearn.model_selection import train_test_split
from unittest.mock import MagicMock, patch

from biobank_project.deep_mtl.models.confounder import BottleneckModel, ConfounderPredictor
from biobank_project.deep_mtl.training import confounder as MOD


@pytest.fixture(name='MixedDataMaker', scope='class')
def mixed_dataset_fixture():
    class MixedDataMaker:
        """Test dataset with regression features and mixed regression and
        classification outcomes, where the regression outcomes are used as a
        confounder variable.
        """
        def __init__(self,
        n_samples=500,
        n_features=30,
        n_classes=5,
        n_regression=3,
        random_state=100):
            self.n_classification = n_classes
            self.n_regression = n_regression
            _, Y = make_multilabel_classification(
                n_samples=n_samples, n_features=n_features,
                n_classes=self.n_classification,
                allow_unlabeled=True, random_state=random_state)
            self.class_Y = pd.DataFrame(
                Y, columns=['class_' + str(i) for i in range(n_classes)])
            X, Y = make_regression(
                n_samples=n_samples, n_features=n_features,
                n_targets=self.n_regression,
                random_state=random_state)
            self.reg_X = pd.DataFrame(
                X, columns=['reg_feature_' + str(i) for i in range(n_features)])
            self.reg_Y = pd.DataFrame(
                Y, columns=['regression_' + str(i) for i in range(n_regression)])

            self.X = self.reg_X
            self.Y = pd.merge(self.class_Y, self.reg_Y, left_index=True, right_index=True)
            self.n_features = len(self.reg_X.columns)
    return MixedDataMaker


class TestConfounderTraining:
    @pytest.fixture
    def dataset(self, MixedDataMaker):
        self.n_samples=500
        self.n_features = 200
        self.n_classes = 3
        # Treat the regression targets as confounder variables
        self.n_regression = 2
        return MixedDataMaker(
            n_samples=self.n_samples, n_features=self.n_features,
            n_classes=self.n_classes, n_regression=self.n_regression)

    @pytest.fixture
    def trainer(self, dataset):
        # Init models
        self.n_bottleneck = 15
        outcome_predictor = BottleneckModel(
            n_features=self.n_features,
            n_bottleneck=self.n_bottleneck,
            n_tasks=self.n_classes)
        confounder_predictor = ConfounderPredictor(
            n_bottleneck=self.n_bottleneck,
            n_confounders=self.n_regression)
        # Specify names of confounder variables
        self.confounder_vars = [
            'regression_' + str(i) for i in range(self.n_regression)]
        self.outcome_vars = dataset.Y.columns[
            ~dataset.Y.columns.isin(self.confounder_vars)].tolist()

        confounder_trainer = MOD.ConfounderTrainer(
            outcome_predictor=outcome_predictor,
            confounder_predictor=confounder_predictor,
            batch_size=self.n_samples,
            confounders=self.confounder_vars)

        # Set training and test data in the trainer
        X_train, X_test, Y_train, Y_test = train_test_split(
            dataset.X, dataset.Y, test_size=0.20)
        confounder_trainer.set_training_data(X_train, Y_train)
        confounder_trainer.set_test_data(X_test, Y_test)
        return confounder_trainer

    @pytest.fixture
    def train_args(self, dataset):
        return {
            'n_epochs': 50,
            'prediction_criterion': nn.BCEWithLogitsLoss(),
            'confounder_criterion': nn.MSELoss(),
            'colnames': self.outcome_vars,
            'confounder_penalty': 1.0
        }

    def test_expected_training_results(self, trainer, train_args, dataset):
        results = trainer.train(**train_args)
        # loss results should have train and test losses per epoch
        assert len(results['losses']['epoch']) == train_args['n_epochs']
        # preds from training should have predictions for all outcomes
        cols_in_preds = results['preds'].columns
        assert set(self.outcome_vars).issubset(cols_in_preds)
        # bottleneck from training should represent the number to bottleneck
        # units
        bottleneck_cols = results['bottleneck'].columns
        bottleneck_cols = bottleneck_cols[bottleneck_cols != 'epoch']
        assert len(bottleneck_cols) == self.n_bottleneck

