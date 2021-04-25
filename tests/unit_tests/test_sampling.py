import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_multilabel_classification


import biobank_project.deep_mtl.sampling as MOD


@pytest.fixture(name='ImbalancedDataMaker', scope='class')
def imbal_dataset_fixture():
    class ImbalancedDataMaker:
        def __init__(self,
        n_samples=100,
        n_features=20,
        n_classes=2,
        weights=[0.99, 0.01],
        random_state=100):
            X, one_label = make_classification(
                n_samples=n_samples, n_features=n_features,
                n_informative=n_features//2, n_classes=n_classes,
                weights=weights, random_state=random_state)
            Y = {
                'imbal_0': one_label,
                'imbal_1': np.random.choice(a=2, size=n_samples, replace=True, p=[0.93, 0.07]),
                'imbal_2': np.random.choice(a=2, size=n_samples, replace=True, p=[0.92, 0.08])
            }
            self.X = pd.DataFrame(X, columns=['feature_' + str(i) for i in range(n_features)])
            self.Y = pd.DataFrame.from_dict(Y)
    return ImbalancedDataMaker


@pytest.fixture(name='BalancedDataMaker', scope='class')
def balanced_dataset_fixture():
    class BalancedDataMaker:
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
    return BalancedDataMaker


@pytest.fixture(scope='class')
def imbalanced_dataset(ImbalancedDataMaker):
    return ImbalancedDataMaker()

@pytest.fixture(scope='class')
def balanced_dataset(BalancedDataMaker):
    return BalancedDataMaker()


class TestMajorityDownsampler:
    @pytest.fixture
    def resampler(self):
        return MOD.MajorityDownsampler(random_state=42)

    def test_resample_downsamples_imbalanced_classes(self, imbalanced_dataset, resampler):
        new_X, new_Y = resampler.resample(
            imbalanced_dataset.X, imbalanced_dataset.Y)
        assert len(new_X) < len(imbalanced_dataset.X)

    def test_resample_doesnt_impact_balanced_dataset(self, balanced_dataset, resampler):
        new_X, new_Y = resampler.resample(
            balanced_dataset.X, balanced_dataset.Y)
        assert all(new_X['feature_0'] == balanced_dataset.X['feature_0'])
