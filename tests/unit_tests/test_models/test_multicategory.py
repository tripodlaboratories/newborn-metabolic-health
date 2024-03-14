"""Tests for multicategory models."""
import random
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_multilabel_classification
import torch

from biobank_project.deep_mtl.models import bottleneck
from biobank_project.deep_mtl.models import multicategory as MOD


# Class-scoped fixtures
@pytest.fixture(name='DataMaker', scope='class')
def dataset_fixture():
    class DataMaker:
        def __init__(self, n_samples=100, n_features=30, n_classes=3):
            self.n_samples = n_samples
            self.n_features = n_features
            self.n_classes = n_classes
            self.unique_categories = ('category_0', 'category_1', 'category_2', 'category_3')

            X, Y = make_multilabel_classification(
                n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                random_state=100)
            self.X = torch.Tensor(X)
            self.Y = torch.Tensor(Y)
            np.random.seed(100)
            self.categories = np.random.choice(self.unique_categories, n_samples)
    return DataMaker


class TestMultiCategoryEnsemble:
    @pytest.fixture
    def dataset(self, DataMaker):
        return DataMaker(n_samples=100, n_features=120, n_classes=3)

    @pytest.fixture
    def base_model_args(self, dataset):
        return dict(
            n_features=dataset.n_features,
            n_tasks=dataset.n_classes,
            n_hidden=100)

    @pytest.fixture
    def model(self, dataset, base_model_args):
        return MOD.MultiCategoryEnsemble(
            categories=dataset.unique_categories,
            base_model=bottleneck.EnsembleNetwork,
            base_model_args=base_model_args
        )

    def test_model_progresses_through_forward_with_expected_shape(self, model, dataset):
        model.train()
        category_outputs, combined_outputs = model(dataset.X, dataset.categories)
        for outputs in (category_outputs, combined_outputs):
            assert outputs.shape[0] == dataset.X.shape[0]
            assert outputs.shape[1] == dataset.Y.shape[1]

    def test_model_calls_only_combined_outputs_with_eval(self, model, dataset):
        model.eval()
        with patch.object(model, 'combined_outputs', return_value=None) as mock_combined:
            outputs = model(dataset.X, dataset.categories)
        mock_combined.assert_called_once()

        with patch.object(model, 'category_specific_outputs', return_value=None) as mock_category:
            outputs = model(dataset.X, dataset.categories)
        mock_category.assert_not_called()

    def test_combined_by_weighted_average(self, model, dataset):
        outputs = model.weighted_average_model_outputs(dataset.X, dataset.categories)
        assert outputs.shape[0] == dataset.X.shape[0]
        assert outputs.shape[1] == dataset.Y.shape[1]



class TestUtilityFunctions:
    @pytest.fixture
    def dataset(self, DataMaker):
        return DataMaker(n_samples=100, n_features=120, n_classes=3)

