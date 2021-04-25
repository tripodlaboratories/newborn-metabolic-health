"""Tests for base models."""
import pytest
import torch
from sklearn.datasets import make_multilabel_classification

from biobank_project.deep_mtl.models import base as MOD


class TestThreeLayerMultiOutput:
    @pytest.fixture(name='DataMaker')
    def dataset_fixture(self):
        class DataMaker:
            def __init__(self, n_samples=100, n_features=30, n_classes=3):
                self.n_samples = n_samples
                self.n_features = n_features
                self.n_classes = n_classes

                X, Y = make_multilabel_classification(
                    n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                    random_state=100)
                self.X = torch.Tensor(X)
                self.Y = torch.Tensor(Y)
        return DataMaker

    @pytest.fixture
    def dataset(self, DataMaker):
        return DataMaker()

    def test_forward_matches_outputs_to_classes_after_forward(self, dataset):
        model = MOD.ThreeLayerMultiOutput(
            n_features=dataset.n_features, n_outputs=dataset.n_classes)
        model_outputs = model.forward(dataset.X)
        assert dataset.n_classes == model_outputs.size()[1]

    def test_class_init_fails_with_incorrect_hidden_layer_spec(self, dataset):
        hidden_layer_spec = {
            'hidden_1': 100,
            'not_to_be_used': 300,
        }
        with pytest.raises(KeyError):
            model = MOD.ThreeLayerMultiOutput(
                n_features=dataset.n_features, n_outputs=dataset.n_classes,
                hidden_layer_spec=hidden_layer_spec)

    def test_class_init_takes_custom_hidden_layer_spec(self, dataset):
        hidden_layer_spec = {
            'hidden_1': 100,
            'hidden_2': 200,
            'hidden_3': 100
        }
        model = MOD.ThreeLayerMultiOutput(
            n_features=dataset.n_features, n_outputs=dataset.n_classes,
            hidden_layer_spec=hidden_layer_spec
        )
        model_outputs = model.forward(dataset.X)
