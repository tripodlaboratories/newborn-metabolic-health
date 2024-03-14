import pandas as pd
import pytest
from sklearn.datasets import make_regression, make_multilabel_classification
from torch import Tensor

from biobank_project.deep_mtl.models import mixed_output as MOD


@pytest.fixture(name='DataMaker', scope='class')
def dataset_fixture():
    class DataMaker:
        def __init__(self,
        n_samples=100,
        n_features=30,
        n_classes=5,
        n_targets=3,
        random_state=100):
            X, Y = make_multilabel_classification(
                n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                allow_unlabeled=True, random_state=random_state)
            self.class_X = pd.DataFrame(
                X, columns=['feature_' + str(i) for i in range(n_features)])
            self.class_Y = pd.DataFrame(
                Y, columns=['class_' + str(i) for i in range(n_classes)])
            X, Y = make_regression(
                n_samples=n_samples, n_features=n_features, n_targets=n_targets,
                random_state=random_state)
            self.reg_X = pd.DataFrame(
                X, columns=['feature_' + str(i) for i in range(n_features)])
            self.reg_Y = pd.DataFrame(
                Y, columns=['regression_' + str(i) for i in range(n_targets)])

            self.X = pd.merge(self.class_X, self.reg_X, left_index=True, right_index=True)
            self.Y = pd.merge(self.class_Y, self.reg_Y, left_index=True, right_index=True)
            self.n_features = len(self.class_X.columns) + len(self.reg_X.columns)
    return DataMaker


class TestThreeLayerMultiOutput:
    @pytest.fixture
    def dataset(self, DataMaker):
        return DataMaker()

    @pytest.fixture
    def model(self, dataset):
        n_features = len(dataset.X.columns)
        n_reg_outputs = len([col for col in dataset.Y.columns if 'reg' in col])
        n_class_outputs = len([col for col in dataset.Y.columns if 'class' in col])
        return MOD.ThreeLayerMultiOutput(
            n_features=n_features, n_class_outputs=n_class_outputs,
            n_reg_outputs=n_reg_outputs)

    def test_model_progression_returns_expected_shapes(self, model, dataset):
        model_output = model.forward(Tensor(dataset.X.values))
        n_samples = dataset.X.shape[0]
        n_features = dataset.n_features
        n_regression = len(dataset.reg_Y.columns)
        n_classification = len(dataset.class_Y.columns)
        assert (n_samples, n_regression) == model_output['regression_out'].shape
        assert (n_samples, n_classification) == model_output['class_out'].shape


class TestEnsemble:
    @pytest.fixture
    def dataset(self, DataMaker):
        return DataMaker()

    @pytest.fixture
    def model(self, dataset):
        n_features = len(dataset.X.columns)
        n_reg_outputs = len([col for col in dataset.Y.columns if 'reg' in col])
        n_class_outputs = len([col for col in dataset.Y.columns if 'class' in col])
        return MOD.EnsembleNetwork(
            n_features=n_features,
            n_class_tasks=n_class_outputs,
            n_reg_tasks=n_reg_outputs)

    def test_model_progression_returns_shapes(self, model, dataset):
        model_output = model.forward(Tensor(dataset.X.values))
        n_samples = dataset.X.shape[0]
        n_features = dataset.n_features
        n_regression = len(dataset.reg_Y.columns)
        n_classification = len(dataset.class_Y.columns)
        assert (n_samples, n_regression) == model_output['regression_out'].shape
        assert (n_samples, n_classification) == model_output['class_out'].shape
