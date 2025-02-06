"""Tests for KFold Cross Validation."""
import pytest
from collections import namedtuple
import numpy as np
import pandas as pd
from sklearn.datasets import (make_classification,
                              make_multilabel_classification,
                              make_regression)
from torch import nn, Tensor
from unittest.mock import MagicMock, patch

from biobank_project.deep_mtl.models import bottleneck, bottleneck_variants
from biobank_project.deep_mtl.training import handlers
from biobank_project.deep_mtl.sampling import MajorityDownsampler
from biobank_project.deep_mtl.training import kfold as MOD

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
        n_samples=150,
        n_features=30,
        n_classes=5,
        random_state=100):
            X, Y = make_multilabel_classification(
                n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                allow_unlabeled=True, random_state=random_state)
            self.X = pd.DataFrame(
                X, columns=['feature_' + str(i) for i in range(n_features)],
                index=[i * 10 for i in range(n_samples)])
            self.Y = pd.DataFrame(
                Y, columns=['label_' + str(i) for i in range(n_classes)],
                index=[i * 10 for i in range(n_samples)])
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
        n_classes=3,
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
class TestRepeatedKFold:
    @pytest.fixture
    def dataset(self, DataMaker):
        n_samples = 500
        n_features = 30
        n_classes = 3
        return DataMaker(n_samples, n_features, n_classes)

    @pytest.fixture
    def test_model(self, dataset, TestModel):
        return TestModel(
            n_inputs=dataset.X.shape[1], n_hidden=100, n_outputs=dataset.Y.shape[1])




    @pytest.fixture
    def training_runner(self, test_model, dataset, mock_wandb_run):
        return handlers.ModelTraining(
            test_model, batch_size=50, shuffle_batch=True, wandb_run=mock_wandb_run)

    @pytest.fixture
    def train_args(self, test_model, dataset):
        return {
            'n_epochs': 5,
            'criterion': nn.BCEWithLogitsLoss(),
            'colnames': dataset.Y.columns,
            'early_stopping_handler': None
        }

    def test_kfold_collects_results_across_all_folds(self, dataset, training_runner, train_args):
        n_folds = 3
        rkf = MOD.RepeatedKFold(
            n_iter=3, n_folds=n_folds, data_X=dataset.X, data_Y=dataset.Y,
            training_handler=training_runner)
        results = rkf.kfold(train_args)
        assert n_folds == len(results['preds']['fold'].unique())
        assert n_folds == len(results['losses']['fold'].unique())

        # Test that indices are kept
        assert all(ix in set(dataset.X.index) for ix in results['preds'].index)

    def test_kfold_properly_splits_data(self, dataset, training_runner, train_args):
        n_folds = 3
        rkf = MOD.RepeatedKFold(
            n_iter=3, n_folds=n_folds, data_X=dataset.X, data_Y=dataset.Y,
            training_handler=training_runner)
        results = rkf.kfold(train_args)
        true_vals = results['true_vals']
        assert len(true_vals.index) == len(true_vals.index.unique())
        assert sorted(dataset.X.index) == sorted(true_vals.index)

        for i in range(n_folds):
            in_fold = true_vals[true_vals.fold == i]
            out_of_fold = true_vals[true_vals.fold != i]
            assert set(in_fold.index).isdisjoint(out_of_fold.index)

    def test_repeated_kfold_collects_results_across_iters(self, dataset, training_runner, train_args):
        n_iter = 3
        n_folds = 3
        rkf = MOD.RepeatedKFold(
            n_iter=n_iter, n_folds=n_folds, data_X=dataset.X,
            data_Y=dataset.Y, training_handler=training_runner)
        results = rkf.repeated_kfold(train_args)
        assert n_iter == len(results['preds']['iter'].unique())
        assert n_iter == len(results['losses']['iter'].unique())
        assert n_iter == len(results['true_vals']['iter'].unique())
        # Test that indices are kept
        assert all(ix in set(dataset.X.index) for ix in results['preds'].index)

    def test_repeated_kfold_resets_as_iters_progress(self, dataset, training_runner, train_args):
        # Idea: Losses in the first and last iters should be close -
        # If the model isn't being reset, losses will continuously decrease across
        # iterations of repeated kfold
        n_iter = 5
        rkf = MOD.RepeatedKFold(
            n_iter=n_iter, n_folds=3, data_X=dataset.X,
            data_Y=dataset.Y, training_handler=training_runner)
        results = rkf.repeated_kfold(train_args)
        loss_df = results['losses']
        first_iter_loss = loss_df[
            (loss_df['epoch'] == 0) &
            (loss_df['fold'] == 0) &
            (loss_df['iter'] == 0)]['train_loss'].item()
        last_iter_num = range(n_iter)[-1]
        last_iter_loss = loss_df[
            (loss_df['epoch'] == 0) &
            (loss_df['fold'] == 0) &
            (loss_df['iter'] == last_iter_num)]['train_loss'].item()
        loss_delta = abs(first_iter_loss - last_iter_loss)
        assert loss_delta <= 0.1

    @patch('biobank_project.deep_mtl.training.handlers.EarlyStopping.evaluate_epoch_loss', autospec=True)
    def test_repeated_kfold_uses_early_stopping_arg_to_call_eval(self, mock_eval, dataset, training_runner, train_args):
        train_args['early_stopping_handler'] = handlers.EarlyStopping(patience=3)

        # Set return value so that training can continue with mock function
        mock_eval.return_value = False
        rkf = MOD.RepeatedKFold(
            n_iter=3, n_folds=3, data_X=dataset.X,
            data_Y=dataset.Y, training_handler=training_runner)
        results = rkf.repeated_kfold(train_args)
        assert mock_eval.called

    def test_repeated_kfold_uses_early_stopping(self, dataset, training_runner, train_args):
        train_args['early_stopping_handler'] = handlers.EarlyStopping(patience=3)
        rkf = MOD.RepeatedKFold(
            n_iter=3, n_folds=3, data_X=dataset.X,
            data_Y=dataset.Y, training_handler=training_runner)
        results = rkf.repeated_kfold(train_args)

    def test_repeated_kfold_can_use_resampler(
            self, ImbalancedDataMaker, TestModel, mock_wandb_run):
        np.random.seed(120)
        dataset = ImbalancedDataMaker()
        model = TestModel(
            n_inputs=dataset.X.shape[1], n_hidden=100, n_outputs=dataset.Y.shape[1])
        training_runner = handlers.ModelTraining(
            model, batch_size=50, shuffle_batch=False, wandb_run=mock_wandb_run)
        train_args = {
            'n_epochs': 5,
            'criterion': nn.BCEWithLogitsLoss(),
            'colnames': dataset.Y.columns,
            'early_stopping_handler': None
        }
        rkf = MOD.RepeatedKFold(
            n_iter=3, n_folds=3, data_X=dataset.X,
            data_Y=dataset.Y, training_handler=training_runner)
        resampler = MajorityDownsampler(random_state=42)
        results = rkf.repeated_kfold(train_args, resampler=resampler)

    def test_repeated_kfold_can_use_validation_data(self, DataMaker, TestModel, mock_wandb_run):
        np.random.seed(80)
        dataset = DataMaker(n_classes=3)
        validation_data = DataMaker(n_classes=3)
        model = TestModel(
            n_inputs=dataset.X.shape[1],
            n_hidden=100,
            n_outputs=dataset.Y.shape[1])
        training_runner = handlers.ModelTraining(
            model, batch_size=250, wandb_run=mock_wandb_run, shuffle_batch=False)
        train_args = {
            'n_epochs': 5,
            'criterion': nn.BCEWithLogitsLoss(),
            'colnames': dataset.Y.columns,
            'early_stopping_handler': None
        }
        rkf = MOD.RepeatedKFold(
            n_iter=10, n_folds=3, data_X=dataset.X,
            data_Y=dataset.Y, training_handler=training_runner,
            X_valid=validation_data.X, Y_valid=validation_data.Y)
        results = rkf.repeated_kfold(train_args)
        assert 'valid_preds' in results


class TestRepeatedKFoldMixedOutput:
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
        return handlers.MixedOutputTraining(
            test_model, batch_size=250, reg_cols=reg_cols,
            class_cols=class_cols, wandb_run=mock_wandb_run)

    @pytest.fixture
    def train_args(self, test_model, dataset):
        return {
            'n_epochs': 5,
            'reg_criterion': nn.MSELoss(),
            'class_criterion': nn.BCEWithLogitsLoss(),
            'early_stopping_handler': None
        }

    def test_repeated_kfold_collects_results_across_iters(self, dataset, training_runner, train_args):
        n_iter = 3
        n_folds = 3
        reg_tasks = [col for col in dataset.Y.columns if 'regression' in col]
        class_tasks = [col for col in dataset.Y.columns if 'class' in col]
        rkf = MOD.RepeatedKFold(
            n_iter=n_iter, n_folds=n_folds, data_X=dataset.X,
            data_Y=dataset.Y, training_handler=training_runner)
        results = rkf.repeated_kfold(
            train_args, mixed_output=True,
            reg_tasks=reg_tasks,
            class_tasks=class_tasks,
            splitting_cols=class_tasks)
        assert n_iter == len(results['preds']['iter'].unique())
        assert n_iter == len(results['losses']['iter'].unique())


class TestKFoldCovariateHandler:
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
        return handlers.CovariateBottleneckTraining(
            test_model, batch_size=50, covariates=dataset.covariates,
            wandb_run=mock_wandb_run)

    @pytest.fixture
    def train_args(self, test_model, dataset):
        return {
            'n_epochs': 5,
            'criterion': nn.BCEWithLogitsLoss(),
            'colnames': list(set(dataset.Y.columns).difference(dataset.covariates)),
            'early_stopping_handler': None,
            'output_training_preds': True
        }

    def test_repeated_kfold_collects_results_across_iters(self, dataset, training_runner, train_args):
        n_iter = 3
        n_folds = 3
        outcome_cols = list(set(dataset.Y.columns).difference(dataset.covariates))
        rkf = MOD.RepeatedKFold(
            n_iter=n_iter, n_folds=n_folds, data_X=dataset.X,
            data_Y=dataset.Y, training_handler=training_runner)
        results = rkf.repeated_kfold(
            train_args,
            splitting_cols=outcome_cols,
            columns_to_score=outcome_cols)
        assert n_iter == len(results['preds']['iter'].unique())
        assert n_iter == len(results['losses']['iter'].unique())


class TestUtils:
    @pytest.fixture
    def epoch_fold_iter(self):
        return {
            'epoch': [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6],
            'fold': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
            'iter': [1] * 18,
        }

    @pytest.fixture
    def preds(self, epoch_fold_iter):
        np.random.seed(0)
        preds = pd.DataFrame.from_dict({
            'task_1': np.random.normal(size=18),
            'task_2': np.random.normal(size=18),
            'epoch': epoch_fold_iter['epoch'],
            'fold': epoch_fold_iter['fold'],
            'iter': epoch_fold_iter['iter']
        })
        return preds

    @pytest.fixture
    def losses(self, epoch_fold_iter):
        losses = pd.DataFrame.from_dict({
            'train_loss': [6.4, 5, 4.2, 3, 3.0, 6.3, 5, 4, 3, 2, 1, 6, 5, 4, 3, 2, 2, 1],
            'test_loss': [6.4, 5, 4, 3, 3.0, 6.3, 5, 4, 3, 2, 1, 6, 5, 4, 3, 2, 2, 1],
            'epoch': epoch_fold_iter['epoch'],
            'fold': epoch_fold_iter['fold'],
            'iter': epoch_fold_iter['iter']
        })
        return losses

    def test_get_lowest_loss_preds_gets_one_pred_per_fold(self, preds, losses):
        filtered_preds = MOD.get_lowest_loss_preds(preds, losses)
        n_folds = len(preds['fold'].unique())
        assert filtered_preds.shape[0] == n_folds
        assert len(filtered_preds['fold'].unique()) == n_folds
