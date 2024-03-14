"""Tests for ensemble models."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_multilabel_classification
import torch

from biobank_project.deep_mtl.models import ensemble as MOD


# Class-scoped fixtures
@pytest.fixture(name='DataMaker', scope='class')
def dataset_fixture():
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


class TestForwardFunctions:
    @pytest.fixture(name='ModelOutputMaker')
    def model_output_fixture(self):
        class ModelOutputMaker:
            def __init__(self, n_models=3, n_ind_outputs=100):
                np.random.seed(100)
                self.n_models = n_models
                self.n_ind_outputs = n_ind_outputs
                self.model_outputs = {
                    'model_' + str(i): torch.Tensor(
                        np.random.normal(
                            size=self.n_ind_outputs))
                for i in range(self.n_models)}
        return ModelOutputMaker

    @pytest.fixture
    def dataset(self, DataMaker):
        return DataMaker(n_samples=100, n_features=120, n_classes=3)

    def test_MultiModelHidden_forward_combines_all_model_outputs(self, DataMaker):
        dataset = DataMaker(n_samples=100, n_features=120, n_classes=3)
        individual_model_outputs = 150
        model = MOD.MultiModelHidden(
            n_combined_input=dataset.n_features, n_models=dataset.n_classes,
            n_outputs=individual_model_outputs)
        model_outputs = model.forward(dataset.X)
        assert individual_model_outputs * dataset.n_classes == model_outputs.size()[1]

    def test_EnsembleNetwork_progresses_through_forward(self, dataset):
        model = MOD.EnsembleNetwork(
            n_features=dataset.n_features, n_tasks=dataset.n_classes)
        model_outputs = model.forward(dataset.X)

    def test_ParallelEnsembleNetwork_progresses_through_forward(self, dataset):
        model = MOD.ParallelEnsembleNetwork(
            n_features=dataset.n_features, n_tasks=dataset.n_classes)
        model_outputs = model.forward(dataset.X)

    def test_SameLayerConnectedFirstInput_outputs_dict_corresponding_to_models(self, dataset):
        model = MOD.SameLayerConnectedFirstInput(
            prev_layer_input=dataset.n_features, n_models=dataset.n_classes)
        model_outputs = model.forward(dataset.X)
        assert dataset.n_classes == len(model_outputs.keys())
