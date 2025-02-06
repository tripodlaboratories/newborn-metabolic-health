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
from sklearn.model_selection import train_test_split
from unittest.mock import MagicMock, patch

from biobank_project.deep_mtl.models import bottleneck, bottleneck_variants
from biobank_project.deep_mtl.training import handlers as MOD
from biobank_project.deep_mtl.training import utils


# Fixtures
@pytest.fixture(name='TestModel', scope='class')
def test_model_fixture():
    class TestModel(nn.Module):
        def __init__(self, n_inputs, n_hidden, n_outputs):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(n_inputs, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_outputs)
            )
        def forward(self, xb):
            return self.layers(xb)
    return TestModel


@pytest.fixture(name='TestMixedModel', scope='class')
def test_mixed_model_fixture():
    class TestMixedModel(nn.Module):
        def __init__(self, n_inputs, n_hidden, n_reg, n_class):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(n_inputs, n_hidden),
                nn.ReLU()
            )

            self.regression_out = nn.Linear(n_hidden, n_reg)
            self.classification_out = nn.Linear(n_hidden, n_class)
        def forward(self, xb):
            hidden_output = self.layers(xb)
            return {
                'regression_out': self.regression_out(hidden_output),
                'class_out': self.classification_out(hidden_output)
            }
    return TestMixedModel


@pytest.fixture(name='DataMaker', scope='class')
def dataset_fixture():
    class DataMaker:
        def __init__(self,
        n_samples=100,
        n_features=30,
        n_classes=5,
        random_state=100):
            X, Y = make_multilabel_classification(
                n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                allow_unlabeled=True, random_state=random_state)
            self.X = pd.DataFrame(X, columns=['feature_' + str(i) for i in range(n_features)])
            self.Y = pd.DataFrame(Y, columns=['label_' + str(i) for i in range(n_classes)])
    return DataMaker


@pytest.fixture(name='ImbalancedDataMaker', scope='class')
def imbalanced_dataset_fixture():
    class ImbalancedDataMaker:
        def __init__(self,
        n_samples=500,
        n_features=30,
        n_classes=2,
        weights=[0.8, 0.2],
        random_state=100):
            X, one_label = make_classification(
                n_samples=n_samples, n_features=n_features,
                n_informative=n_features//2, n_classes=n_classes,
                weights=weights, random_state=random_state)
            Y = {
                'imbal_0': one_label,
                'imbal_1': np.random.choice(a=2, size=n_samples, replace=True, p=[0.8, 0.2]),
                'imbal_2': np.random.choice(a=2, size=n_samples, replace=True, p=[0.8, 0.2])
            }
            self.X = pd.DataFrame(X, columns=['feature_' + str(i) for i in range(n_features)])
            self.Y = pd.DataFrame.from_dict(Y)
    return ImbalancedDataMaker


@pytest.fixture(name='MixedDataMaker', scope='class')
def mixed_dataset_fixture():
    class MixedDataMaker:
        def __init__(self,
        n_samples=500,
        n_features=30,
        n_classes=5,
        n_targets=3,
        random_state=100):
            self.n_classification = n_classes
            self.n_regression = n_targets
            X, Y = make_multilabel_classification(
                n_samples=n_samples, n_features=n_features,
                n_classes=self.n_classification,
                allow_unlabeled=True, random_state=random_state)
            self.class_X = pd.DataFrame(
                X, columns=['class_feature_' + str(i) for i in range(n_features)])
            self.class_Y = pd.DataFrame(
                Y, columns=['class_' + str(i) for i in range(n_classes)])
            X, Y = make_regression(
                n_samples=n_samples, n_features=n_features,
                n_targets=self.n_regression,
                random_state=random_state)
            self.reg_X = pd.DataFrame(
                X, columns=['reg_feature_' + str(i) for i in range(n_features)])
            self.reg_Y = pd.DataFrame(
                Y, columns=['regression_' + str(i) for i in range(n_targets)])

            self.X = pd.merge(self.class_X, self.reg_X, left_index=True, right_index=True)
            self.Y = pd.merge(self.class_Y, self.reg_Y, left_index=True, right_index=True)
            self.n_features = len(self.class_X.columns) + len(self.reg_X.columns)
    return MixedDataMaker


@pytest.fixture
def mock_wandb_run():
    # Avoid external dependency, and requirement to call wandb.init() in tests
    wandb_run = MagicMock()
    wandb_run.log = MagicMock()
    return wandb_run


# Test Classes
class TestEarlyStopping:
    @pytest.fixture
    def earlystop(self):
        return MOD.EarlyStopping(patience=5, min_delta=0.0)

    def test_early_stopping_raises_error_on_invalid_patience(self):
        with pytest.raises(ValueError):
            MOD.EarlyStopping(patience=-1)

    def test_early_stopping_raises_error_on_neg_mindelta(self):
        with pytest.raises(ValueError):
            MOD.EarlyStopping(patience=5, min_delta=-0.1)

    def test_evaluate_epoch_loss_processes_new_lower_loss(self, earlystop):
        earlystop.lowest_loss = 0.5
        current_loss = 0.2
        earlystop_criterion = earlystop.evaluate_epoch_loss(current_loss)
        assert earlystop_criterion is False
        assert earlystop.epochs_without_improvement == 0

    def test_evaluate_epoch_loss_accounts_for_patience_without_improvement(self, earlystop):
        earlystop.lowest_loss = 0.5
        earlystop.epochs_without_improvement = 3
        current_loss = 0.6
        earlystop_criterion = earlystop.evaluate_epoch_loss(current_loss)
        assert earlystop_criterion is False

    def test_evaluate_epoch_loss_sets_earlystop_epochs_greater_than_patience(self, earlystop):
        earlystop.lowest_loss = 0.5
        earlystop.epochs_without_improvement = 5
        current_loss = 0.6
        earlystop_criterion = earlystop.evaluate_epoch_loss(current_loss)
        assert earlystop_criterion is True


class TestModelTraining:
    @pytest.fixture
    def dataset(self, DataMaker):
        n_samples = 500
        n_features = 30
        n_classes = 3
        random_state = 100
        return DataMaker(n_samples, n_features, n_classes, random_state)

    @pytest.fixture
    def validation_dataset(self, DataMaker):
        n_samples = 500
        n_features = 30
        n_classes = 3
        random_state = 100
        return DataMaker(n_samples, n_features, n_classes, random_state)

    @pytest.fixture
    def test_model(self, dataset, TestModel):
        return TestModel(
            n_inputs=dataset.X.shape[1], n_hidden=100, n_outputs=dataset.Y.shape[1])


    @pytest.fixture
    def training_runner(
        self,
        test_model,
        dataset,
        mock_wandb_run):
        return MOD.ModelTraining(test_model, batch_size=50, wandb_run=mock_wandb_run)

    @pytest.fixture
    def train_args(self, test_model, dataset):
        return {
            'n_epochs': 50,
            'criterion': nn.BCEWithLogitsLoss(),
            'colnames': dataset.Y.columns,
            'early_stopping_handler': None,
            'output_training_preds': True
        }

    def test_set_training_data_populates_loader(self, training_runner, dataset):
        training_runner.set_training_data(dataset.X, dataset.Y)
        assert training_runner.train_loader is not None
        assert training_runner.test_loader is None

    def test_set_test_data_populates_loader(self, training_runner, dataset):
        training_runner.set_test_data(dataset.X, dataset.Y)
        assert training_runner.test_loader is not None
        assert training_runner.train_loader is None

    def test_train_records_results_for_all_epochs(self, training_runner, dataset):
        criterion = nn.BCEWithLogitsLoss()
        n_epochs = 5
        colnames = dataset.Y.columns
        X_train, X_test, Y_train, Y_test = train_test_split(dataset.X, dataset.Y)

        training_runner.set_training_data(X_train, Y_train)
        training_runner.set_test_data(X_test, Y_test)
        results = training_runner.train(
            n_epochs=n_epochs, criterion=criterion, colnames=colnames)
        preds = results['preds']
        losses = results['losses']

        assert n_epochs == len(preds['epoch'].unique())
        assert n_epochs == len(losses['epoch'])

    def test_model_can_be_trained_after_reset(self, training_runner, dataset):
        criterion = nn.BCEWithLogitsLoss()
        n_epochs = 5
        colnames = dataset.Y.columns
        X_train, X_test, Y_train, Y_test = train_test_split(dataset.X, dataset.Y)

        training_runner.set_training_data(X_train, Y_train)
        training_runner.set_test_data(X_test, Y_test)
        results = training_runner.train(
            n_epochs=n_epochs, criterion=criterion, colnames=colnames)
        training_runner.reset_model()
        results = training_runner.train(
            n_epochs=n_epochs, criterion=criterion, colnames=colnames)

    def test_model_training_uses_early_stopping(self, training_runner, dataset, train_args):
        MOD.utils.seed_torch(100)
        X_train, X_test, Y_train, Y_test = train_test_split(
            dataset.X, dataset.Y, random_state=100)
        training_runner.set_training_data(X_train, Y_train)
        training_runner.set_test_data(X_test, Y_test)
        train_args['early_stopping_handler'] = MOD.EarlyStopping(patience=1)
        results = training_runner.train(**train_args)
        preds = results['preds']
        assert preds['epoch'].max() < train_args['n_epochs']

    def test_model_training_losses_correspond_to_auc_gain_in_training_set(self, training_runner, dataset, train_args):
        utils.seed_torch(100)
        train_args['n_epochs'] = 25
        X_train, X_test, Y_train, Y_test = train_test_split(
            dataset.X, dataset.Y, random_state=100)
        training_runner.set_training_data(X_train, Y_train)
        training_runner.set_test_data(X_test, Y_test)
        results = training_runner.train(**train_args)
        preds = results['train_preds']
        final_epoch_preds = preds[preds['epoch'] == preds['epoch'].max()]
        scores = utils.score_predictions(final_epoch_preds, Y_train, columns_to_score=Y_train.columns)

        # Model should at least be able to fit the training set
        assert all(scores['auroc'] > 0.70)

    def test_model_training_can_use_class_weights(self, training_runner, dataset, train_args):
        pos_weights = dataset.Y.apply(utils.get_pos_weight)
        train_args['criterion'] = nn.BCEWithLogitsLoss(pos_weight=Tensor(pos_weights.values))
        train_args['n_epochs'] = 5
        utils.seed_torch(100)
        X_train, X_test, Y_train, Y_test = train_test_split(
            dataset.X, dataset.Y, random_state=100)
        training_runner.set_training_data(X_train, Y_train)
        training_runner.set_test_data(X_test, Y_test)
        results = training_runner.train(**train_args)

    def test_validation_dataloader(self, training_runner, dataset, train_args, validation_dataset):
        utils.seed_torch(100)
        train_args['n_epochs'] = 25
        X_train, X_test, Y_train, Y_test = train_test_split(
            dataset.X, dataset.Y, random_state=100)
        training_runner.set_training_data(X_train, Y_train)
        training_runner.set_test_data(X_test, Y_test)
        training_runner.set_validation_data(
            validation_dataset.X, validation_dataset.Y)
        results = training_runner.train(**train_args)

        assert 'valid_loss' in results['losses'].columns
        assert 'valid_preds' in results.keys()


class TestMixedOutput:
    @pytest.fixture
    def dataset(self, MixedDataMaker):
        return MixedDataMaker()

    @pytest.fixture
    def test_model(self, dataset, TestMixedModel):
        return TestMixedModel(
            n_inputs=dataset.X.shape[1], n_hidden=100,
            n_class=dataset.n_classification,
            n_reg=dataset.n_regression)

    @pytest.fixture
    def training_runner(self, test_model, dataset, mock_wandb_run):
        reg_cols = [col for col in dataset.Y.columns if 'regression' in col]
        class_cols = [col for col in dataset.Y.columns if 'class' in col]
        return MOD.MixedOutputTraining(
            test_model, batch_size=50, reg_cols=reg_cols,
            class_cols=class_cols, wandb_run=mock_wandb_run)

    def test_set_training_data_populates_loader(self, training_runner, dataset):
        training_runner.set_training_data(dataset.X, dataset.Y)
        assert training_runner.train_loader is not None
        assert training_runner.test_loader is None

    def test_set_test_data_populates_loader(self, training_runner, dataset):
        training_runner.set_test_data(dataset.X, dataset.Y)
        assert training_runner.test_loader is not None
        assert training_runner.train_loader is None

    def test_train_outputs_preds_and_losses_per_epoch(self, training_runner, dataset):
        n_epochs = 5
        colnames = dataset.Y.columns
        X_train, X_test, Y_train, Y_test = train_test_split(dataset.X, dataset.Y)

        training_runner.set_training_data(X_train, Y_train)
        training_runner.set_test_data(X_test, Y_test)
        results = training_runner.train(
            n_epochs=n_epochs,
            reg_criterion=nn.MSELoss(),
            class_criterion=nn.BCEWithLogitsLoss())
        preds = results['preds']
        losses = results['losses']
        assert n_epochs == len(preds['epoch'].unique())
        assert n_epochs == len(losses['epoch'])


class TestBottleneckHandler:
    @pytest.fixture(name='TestBottleneckModel')
    def test_model_fixture(self):
        class TestBottleneckModel(nn.Module):
            def __init__(self, n_inputs, n_hidden, n_bottleneck, n_outputs):
                super().__init__()
                self.n_bottleneck = n_bottleneck

                self.layers = nn.Sequential(
                    nn.Linear(n_inputs, n_hidden),
                    nn.ReLU(),
                )

                self.bottleneck = nn.Sequential(
                    nn.Linear(n_hidden, n_bottleneck),
                    nn.ReLU()
                )
                self.out = nn.Linear(n_bottleneck, n_outputs)

            def forward(self, xb, return_bottleneck=False):
                xb = self.layers(xb)
                bottleneck = self.bottleneck(xb)
                model_output = self.out(bottleneck)
                if return_bottleneck is True:
                    return model_output, bottleneck
                else:
                    return model_output
        return TestBottleneckModel

    @pytest.fixture
    def dataset(self, DataMaker):
        return DataMaker()

    @pytest.fixture
    def validation_dataset(self, DataMaker):
        n_samples = 500
        n_features = 30
        n_classes = 5
        random_state = 100
        return DataMaker(n_samples, n_features, n_classes, random_state)

    @pytest.fixture
    def test_model(self, dataset, TestBottleneckModel):
        return TestBottleneckModel(
            n_inputs=dataset.X.shape[1],
            n_hidden=100,
            n_bottleneck=5,
            n_outputs=dataset.Y.shape[1])

    @pytest.fixture
    def training_runner(self, test_model, dataset, mock_wandb_run):
        return MOD.BottleneckModelTraining(test_model, wandb_run=mock_wandb_run, batch_size=50)

    @pytest.fixture
    def train_args(self, test_model, dataset):
        return {
            'n_epochs': 50,
            'criterion': nn.BCEWithLogitsLoss(),
            'colnames': dataset.Y.columns,
            'early_stopping_handler': None,
            'output_training_preds': True
        }

    def test_handler_returns_bottleneck_results(self, training_runner, dataset, test_model, train_args):
        MOD.utils.seed_torch(100)
        X_train, X_test, Y_train, Y_test = train_test_split(
            dataset.X, dataset.Y, random_state=100)
        training_runner.set_training_data(X_train, Y_train)
        training_runner.set_test_data(X_test, Y_test)
        results = training_runner.train(**train_args)
        bottleneck = results['bottleneck']
        bottleneck_cols = [col for col in bottleneck.columns if 'bottle' in col]
        assert len(bottleneck_cols) == test_model.n_bottleneck

    def test_validation_dataloader(self, training_runner, dataset, train_args, validation_dataset):
        utils.seed_torch(100)
        train_args['n_epochs'] = 25
        X_train, X_test, Y_train, Y_test = train_test_split(
            dataset.X, dataset.Y, random_state=100)
        training_runner.set_training_data(X_train, Y_train)
        training_runner.set_test_data(X_test, Y_test)
        training_runner.set_validation_data(
            validation_dataset.X, validation_dataset.Y)
        results = training_runner.train(**train_args)

        assert 'valid_loss' in results['losses'].columns
        assert 'valid_preds' in results.keys()


class TestCovariateBottleneckHandler:
    @pytest.fixture
    def dataset(self, DataMaker):
        test_data = DataMaker()
        TestDataset = namedtuple('TestData', ['X', 'Y', 'covariates'])
        return TestDataset(
            X=test_data.X,
            Y=test_data.Y,
            covariates=['label_3', 'label_4'])

    @pytest.fixture
    def test_model(self, dataset):
        return bottleneck_variants.CovariateEnsemble(
            n_features=dataset.X.shape[1],
            n_tasks=dataset.Y.shape[1] - len(dataset.covariates),
            n_hidden=50,
            n_bottleneck=5,
            n_covariates=len(dataset.covariates)
            )

    @pytest.fixture
    def training_runner(self, test_model, dataset, mock_wandb_run):
        return MOD.CovariateBottleneckTraining(
            test_model, batch_size=50, covariates=dataset.covariates,
            wandb_run=mock_wandb_run)

    @pytest.fixture
    def train_args(self, test_model, dataset):
        return {
            'n_epochs': 50,
            'criterion': nn.BCEWithLogitsLoss(),
            'colnames': list(set(dataset.Y.columns).difference(dataset.covariates)),
            'early_stopping_handler': None,
            'output_training_preds': True
        }

    def test_handler_returns_bottleneck_results(self, training_runner, dataset, test_model, train_args):
        MOD.utils.seed_torch(100)
        X_train, X_test, Y_train, Y_test = train_test_split(
            dataset.X, dataset.Y, random_state=100)
        training_runner.set_training_data(X_train, Y_train)
        training_runner.set_test_data(X_test, Y_test)
        results = training_runner.train(**train_args)
        assert 'bottleneck' in results
