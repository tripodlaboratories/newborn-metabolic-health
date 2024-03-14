"""Re-sampling modules for model training."""
from abc import ABC, abstractmethod
import warnings

import numpy as np
import pandas as pd
from sklearn.utils import resample


from biobank_project.deep_mtl.training.utils import multilabel_to_single_array


class Resampler(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def resample(self, data_X, data_Y):
        pass


class MajorityDownsampler(Resampler):
    def __init__(self, random_state: int=None):
        self.random_state = random_state

    def resample(self, data_X: pd.DataFrame, data_Y: pd.DataFrame):
        encoded_y = multilabel_to_single_array(data_Y)
        unique_y = np.unique(encoded_y)
        majority_class = unique_y[np.argmax(np.bincount(encoded_y))]

        # Split data into majority and other classes
        majority_indices = np.where(encoded_y == majority_class)
        other_indices = np.where(encoded_y != majority_class)
        majority_X = data_X.iloc[majority_indices]
        majority_Y = data_Y.iloc[majority_indices]
        other_X = data_X.iloc[other_indices]
        other_Y = data_Y.iloc[other_indices]

        if len(majority_X) <= len(other_X):
            # Can't downsample majority class in this case
            return data_X, data_Y

        downsamp_X, downsamp_Y = resample(
            majority_X, majority_Y, replace=False,
            n_samples=len(other_X), random_state=self.random_state)
        new_X = pd.concat([downsamp_X, other_X], axis=0).sample(
            frac=1, random_state=self.random_state)
        new_Y = pd.concat([downsamp_Y, other_Y], axis=0).sample(
            frac=1, random_state=self.random_state)
        return new_X, new_Y


