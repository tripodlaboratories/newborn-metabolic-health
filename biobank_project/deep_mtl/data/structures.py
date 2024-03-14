"""Custom Data Structures"""
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor


class IndexedTensorDataset(TensorDataset):
    def __init__(self, *tensors, reference_index=None):
        super().__init__(*tensors)
        if reference_index is not None:
            assert len(reference_index) == tensors[0].size(0)
        self.reference_index = reference_index

    def __getitem__(self, index):
        if self.reference_index is not None:
            return_index = self.reference_index[index]
        else:
            return_index = index

        return return_index, tuple(tensor[index] for tensor in self.tensors)
