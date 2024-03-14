import pytest
import pandas as pd
from sklearn.datasets import make_multilabel_classification
from torch import Tensor
from torch.utils.data import DataLoader


from biobank_project.deep_mtl.data import structures as MOD


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
            self.X = pd.DataFrame(
                X, columns=['feature_' + str(i) for i in range(n_features)],
                index=['sample_' + str(i) for i in range(n_samples)])
            self.Y = pd.DataFrame(
                Y, columns=['label_' + str(i) for i in range(n_classes)],
                index=['sample_' + str(i) for i in range(n_samples)])
    return DataMaker

class TestCustomDataset:
    @pytest.fixture
    def data(self, DataMaker):
        return DataMaker()

    @pytest.fixture
    def indexed_dataset(self, data):
        X_tensor, Y_tensor = Tensor(data.X.values), Tensor(data.Y.values)
        return MOD.IndexedTensorDataset(X_tensor, Y_tensor, reference_index=data.X.index)

    @pytest.fixture
    def loader(self, indexed_dataset):
        return DataLoader(indexed_dataset, batch_size=25, shuffle=True)

    def test_custom_dataset_keeps_indices_after_iter_in_loader(self, data, loader):
        iterated_x = []
        iterated_y = []
        for ix, (xb, yb) in loader:
            iterated_x.append(pd.DataFrame(
                xb.numpy(), columns=data.X.columns, index=ix))
            iterated_y.append(pd.DataFrame(
                yb.numpy(), columns=data.Y.columns, index=ix))

        all_x = pd.concat(iterated_x)
        all_y = pd.concat(iterated_y)
        assert all(ix in set(data.X.index) for ix in all_x.index)
        assert all(ix in set(data.Y.index) for ix in all_y.index)
