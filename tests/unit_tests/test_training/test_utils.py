"""Test training utility functions"""
import pytest
import numpy as np
import pandas as pd

from sklearn.datasets import (make_classification,
                              make_multilabel_classification)

from biobank_project.deep_mtl.training import utils as MOD


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


class TestUtilityFunctions:
    @pytest.fixture
    def dataset(self, DataMaker):
        n_samples = 100
        n_features = 30
        n_classes = 5
        return DataMaker(n_samples, n_features, n_classes)

    @pytest.fixture
    def preds(self, dataset):
        n_classes = len(dataset.Y.columns)
        n_samples = dataset.Y.shape[0]
        return pd.DataFrame.from_dict(
            {'label_' + str(i): np.random.uniform(size=n_samples) for i in range(n_classes)}
        )

    def test_multilabel_to_single_array_returns_array_of_n_samples(self, dataset):
        output_y = MOD.multilabel_to_single_array(dataset.Y)
        assert dataset.Y.shape[0] == len(output_y)

    def test_get_pos_weights_are_greater_for_sample_imbalance(self, dataset):
        balanced_col = pd.Series(
            np.random.choice(a=2, size=250, replace=True, p=[0.5, 0.5]))
        bal_class_weight = MOD.get_pos_weight(balanced_col)
        imbalanced_col = pd.Series(
            np.random.choice(a=2, size=250, replace=True, p=[0.85, 0.15]))
        imbal_class_weight = MOD.get_pos_weight(imbalanced_col)
        assert imbal_class_weight > bal_class_weight

    def test_score_predictions_returns_scores_for_all_cols(self, preds, dataset):
        columns_to_score = [col for col in dataset.Y.columns if 'label' in col]
        scores = MOD.score_predictions(
            preds=preds, true_values=dataset.Y,
            columns_to_score=columns_to_score)
        assert len(scores['task'] == len(columns_to_score))

    def test_dfs_to_datastructs_can_take_variable_num_of_df_args(self, DataMaker):
        dataset_1 = DataMaker()
        dataset_2 = DataMaker()
        dataset, dataloader = MOD.df_to_datastructs(
            dataset_1.X, dataset_1.Y, dataset_2.X, shuffle=False, batch_size=100)
