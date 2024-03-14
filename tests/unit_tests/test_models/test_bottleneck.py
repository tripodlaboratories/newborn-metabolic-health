"""Tests for bottleneck models."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_multilabel_classification
import torch

from biobank_project.deep_mtl.models import bottleneck as MOD
from biobank_project.deep_mtl.models import bottleneck_variants as MOD_VARIANTS


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
    @pytest.fixture
    def dataset(self, DataMaker):
        return DataMaker(n_samples=100, n_features=120, n_classes=3)

    def test_ThreeLayerBottleneck_progresses_through_forward(self, dataset):
        model = MOD.ThreeLayerBottleneck(
            n_features=dataset.n_features,
            n_outputs=dataset.n_classes,
            n_bottleneck=10)
        model_outputs = model.forward(dataset.X)

    def test_ThreeLayerBottleneck_returns_expected_bottleneck_shape(self, dataset):
        n_bottleneck = 10
        model = MOD.ThreeLayerBottleneck(
            n_features=dataset.n_features,
            n_outputs=dataset.n_classes,
            n_bottleneck=n_bottleneck)
        model_outputs, bottleneck = model.forward(dataset.X, return_bottleneck=True)
        assert bottleneck.shape[1] == n_bottleneck

    def test_EnsembleNetwork_progresses_through_forward(self, dataset):
        model = MOD.EnsembleNetwork(
            n_features=dataset.n_features,
            n_tasks=dataset.n_classes,
            n_bottleneck=10
        )
        model_outputs = model.forward(dataset.X)

    def test_EnsembleNetwork_returns_expected_bottleneck_shape(self, dataset):
        n_bottleneck = 10
        model = MOD.EnsembleNetwork(
            n_features=dataset.n_features,
            n_tasks=dataset.n_classes,
            n_bottleneck=n_bottleneck)
        model_outputs, bottleneck = model.forward(dataset.X, return_bottleneck=True)
        assert bottleneck.shape[1] == n_bottleneck

    def test_ParallelEnsembleNetwork_progresses_through_forward(self, dataset):
        model = MOD.ParallelEnsembleNetwork(
            n_features=dataset.n_features,
            n_tasks=dataset.n_classes,
            n_bottleneck=10
        )
        model_outputs = model.forward(dataset.X)

    def test_ParallelEnsembleNetwork_returns_expected_bottleneck_shape(self, dataset):
        n_bottleneck = 5
        model = MOD.ParallelEnsembleNetwork(
            n_features=dataset.n_features,
            n_tasks=dataset.n_classes,
            n_bottleneck=n_bottleneck)
        model_outputs, bottleneck = model.forward(dataset.X, return_bottleneck=True)
        assert bottleneck.shape[1] == n_bottleneck

    def test_EnsembleCovariateBottleneck_returns_outputs(self, dataset):
        # Treat one of the tasks as a covariate for the bottleneck
        covariate = dataset.Y[:, 2]
        tasks = dataset.Y[:, 0:2]
        model = MOD_VARIANTS.CovariateEnsemble(
            n_features=dataset.n_features,
            n_tasks=dataset.n_classes - 1,
            n_bottleneck=5)
        model_outputs = model.forward(dataset.X, covariate)
        assert model_outputs.shape[1] == tasks.shape[1]

    def test_AdditiveCovariate_returns_outputs(self, dataset):
        covariate = dataset.Y[:, 2]
        tasks = dataset.Y[:, 0:2]
        model = MOD_VARIANTS.AdditiveCovariate(
            n_features=dataset.n_features,
            n_tasks=dataset.n_classes - 1,
            n_bottleneck=5)
        model_outputs = model.forward(dataset.X, covariate)
        assert model_outputs.shape[1] == tasks.shape[1]
