"""Multicategory ensemble models."""
import math

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset


from biobank_project.deep_mtl.models import base, ensemble
from biobank_project.deep_mtl.models.bottleneck import EnsembleNetwork


class MultiCategoryEnsemble(pl.LightningModule):
    """
    Architecture that trains an independent model for each given category.
    """
    def __init__(
        self,
        categories: list,
        base_model,
        base_model_args: dict,
        simple_average_combine: bool=True,
        importance_ratio: float=None,
        learn_model_weights: bool=False,
        pos_weight_for_loss=None):
        """
        args:
            categories: known number of categories
            base_model: model to use per category
            base_model_args: dict of arguments to use in init for base model
            importance_ratio: how much weight should be given to matching
                category model prediction / non-category model prediction
        """
        super().__init__()

        self.category_models = nn.ModuleDict(
            {c: base_model(**base_model_args) for c in sorted(categories)})

        # Model average by weights
        if simple_average_combine is True and learn_model_weights is False:
            self.simple_average_combine = True
            self.learn_model_weights = False
            if importance_ratio is None:
                self.importance_ratio = len(self.category_models)

        # Learned weight parameters
        if learn_model_weights is True and simple_average_combine is False:
            self.learn_model_weights = True
            self.simple_average_combine = False
            self.model_weights = nn.parameter.Parameter(
                torch.Tensor(
                    len(self.category_models),
                    len(self.category_models)))
            self.model_weights = nn.init.kaiming_uniform_(
                self.model_weights, a=math.sqrt(5))

        # Other attributes
        self.pos_weight_for_loss = pos_weight_for_loss

    def get_outputs_from_all_models(self, xb):
        all_model_preds = []
        for model in self.category_models.values():
            model_preds = model(xb)
            all_model_preds.append(model_preds)
        return torch.stack(all_model_preds, dim=2)

    def combined_outputs(self, xb, batch_categories):
        """Make combined predictions from all category-specific models.
        """
        if self.simple_average_combine is True:
            preds = self.weighted_average_model_outputs(xb, batch_categories)
        elif self.learn_model_weights is True:
            preds = self.learned_model_weights(xb, batch_categories)
        else:
            raise ValueError('simple_average_combine or learn_model_weights must be specified.')
        return preds

    def category_specific_outputs(self, xb, batch_categories):
        """Use only predictions from the corresponding category-specific model."""
        category_one_hot = torch.Tensor(
            pd.get_dummies(batch_categories).values
            )
        all_model_preds = self.get_outputs_from_all_models(xb)

        # Use one-hot encoding as a filtering matrix for predictions
        # First implementation using one-hot only
        weighted_preds = all_model_preds * category_one_hot.unsqueeze(1)
        weighted_preds = weighted_preds.sum(dim=-1)
        return weighted_preds

    def weighted_average_model_outputs(self, xb, batch_categories):
        category_one_hot = torch.Tensor(
            pd.get_dummies(batch_categories).values
            )
        off_model_filter = torch.ones(category_one_hot.shape) - category_one_hot
        all_model_preds = self.get_outputs_from_all_models(xb)

        # Set up weighted average
        weight_matrix = category_one_hot * self.importance_ratio + off_model_filter
        # Keep weight matrix maxed at 1 for numerical considerations
        weight_matrix /= weight_matrix.max(1, keepdim=True)[0]
        weighted_preds = all_model_preds * weight_matrix.unsqueeze(1)
        weighted_preds = weighted_preds.mean(dim=-1)
        return weighted_preds

    def learned_model_weights(self, xb, batch_categories):
        all_model_preds = []
        category_one_hot = torch.Tensor(
            pd.get_dummies(batch_categories).values
            )
        self.get_outputs_from_all_models(xb)

        # Select corresponding weights for each sample
        weight_matrix = torch.mm(category_one_hot, self.model_weights)
        weighted_preds = all_model_preds * weight_matrix.unsqueeze(1)
        weighted_preds = weighted_preds.mean(dim=-1)
        return weighted_preds

    def forward(self, xb, batch_categories: np.array):
        if self.training is True:
            # First pass: make category-specific predictions from each model
            category_outputs = self.category_specific_outputs(
                xb, batch_categories)

            # Second pass: all models make predictions that are combined
            # NOTE: may only be useful if weights are a learned parameter
            combined_outputs_all_models = self.combined_outputs(xb, batch_categories)
            return category_outputs, combined_outputs_all_models

        else:
            combined_outputs_all_models = self.combined_outputs(xb, batch_categories)
            return combined_outputs_all_models

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        (x, y, batch_categories), ref_index = batch
        batch_categories = np.array(batch_categories)
        category_outputs, combined_outputs = self(x, batch_categories)
        loss = F.binary_cross_entropy_with_logits(
            category_outputs, y, pos_weight=self.pos_weight_for_loss)
        if self.learn_model_weights is True:
            loss += F.binary_cross_entropy_with_logits(
                combined_outputs, y, pos_weight=self.pos_weight_for_loss)
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (x, y, batch_categories), ref_index = batch
        batch_categories = np.array(batch_categories)
        combined_outputs = self(x, batch_categories)
        loss = F.binary_cross_entropy_with_logits(
            combined_outputs, y, pos_weight=self.pos_weight_for_loss)
        self.log('val_loss', loss, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        (x, y, batch_categories), ref_index = batch
        batch_categories = np.array(batch_categories)
        combined_outputs = self(x, batch_categories)
        loss = F.binary_cross_entropy_with_logits(
            combined_outputs, y, pos_weight=self.pos_weight_for_loss)
        self.log('holdout_test_loss', loss, logger=True)
        return loss


class MultiCategoryDataset(TensorDataset):
    def __init__(self, tensors, reference_index=None):
        super().__init__()
        if reference_index is not None:
            assert len(reference_index) == tensors[0].size(0)
        self.reference_index = reference_index
        self.tensors = tensors

    def __getitem__(self, index):
        if self.reference_index is not None:
            return_index = self.reference_index[index]
        else:
            return_index = index
        return tuple(tensor[index] for tensor in self.tensors), return_index

