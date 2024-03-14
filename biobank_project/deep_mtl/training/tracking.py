"""Data structures for tracking outputs and bookkeeping in model training."""
import numpy as np
import pandas as pd
import torch


class TrainingTracker:
    """Data structure for keeping results over epochs."""
    def __init__(self):
        # Expected to contain lists of dataframes
        self.losses = []
        self.train = OutputTracker(track_losses=False)
        self.test = OutputTracker(track_losses=False)
        self.validation = OutputTracker(track_losses=False)


class EpochTracker:
    """Data structure for results within an epoch."""
    def __init__(self, epoch: int):
        self.epoch = epoch
        self.train = OutputTracker()
        self.test = OutputTracker()
        self.validation = OutputTracker()

    def summarize_losses(self) -> pd.DataFrame:
        epoch_losses = {
            'train_loss': [np.mean(self.train.losses)],
            'test_loss': [np.mean(self.test.losses)]}
        if len(self.validation.losses) > 0:
            epoch_losses['valid_loss'] = [np.mean(self.validation.losses)]
        current_epoch_losses = pd.DataFrame.from_dict(epoch_losses)
        current_epoch_losses['epoch'] = self.epoch
        return current_epoch_losses

    def summarize_preds(self, output_tracker, colnames) -> pd.DataFrame:
        current_epoch_preds = pd.DataFrame(
            torch.cat(output_tracker.preds, dim=0).data.numpy(),
            columns=colnames, index=np.concatenate(output_tracker.index))
        current_epoch_preds['epoch'] = self.epoch
        return current_epoch_preds

    def summarize_bottleneck(self, output_tracker) -> pd.DataFrame:
        bottleneck = torch.cat(output_tracker.bottleneck, dim=0).data.numpy()
        n_bottleneck = bottleneck.shape[1]
        current_epoch_bottleneck = pd.DataFrame(
            bottleneck,
            columns=['bottleneck_unit_' + str(i) for i in range(n_bottleneck)],
            index=np.concatenate(output_tracker.index))
        current_epoch_bottleneck['epoch'] = self.epoch
        return current_epoch_bottleneck

    def gather_true_vals(self, output_tracker, colnames) -> pd.DataFrame:
        return pd.DataFrame(
            torch.cat(output_tracker.true_vals, dim=0).data.numpy(),
            columns=colnames,
            index=np.concatenate(output_tracker.index))


class OutputTracker:
    """Data structure for tracking model training outputs."""
    def __init__(self, track_losses=True):
        if track_losses is True:
            # On the highest level (tracking over epochs), train and test
            # losses are combined, negating the need for an additional tracker
            self.losses = []

        # Lists of arrays, tensors within epoch and lists of dataframes over
        # epochs
        self.index = []
        self.preds = []
        self.bottleneck = []
        self.true_vals = []

