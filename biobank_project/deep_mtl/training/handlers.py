"""Module for training deep MTL models."""
import copy
import os
import random
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim, as_tensor
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from biobank_project.deep_mtl.training import utils


class EarlyStopping:
    """Class for evaluating early stopping during model training.
    """
    def __init__(self, patience: int, min_delta: float=0.0):
        """
        args:
            patience: number of epochs without improvement before stopping
            min_delta: padding for score used in evaluating improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        if patience < 1:
            raise ValueError('Argument patience should be positive integer.')
        if min_delta < 0.0:
            raise ValueError('Argument min_delta should not be a negative number.')

        self.lowest_loss = None
        self.epochs_without_improvement = 0

    def reset(self):
        self.lowest_loss = None

    def evaluate_epoch_loss(self, current_loss: float) -> bool:
        """Evaluate the loss of the current epoch in context of lowest loss of all epochs.
        """
        if self.lowest_loss is None:
            # i.e., the current epoch is the first epoch
            self.lowest_loss = current_loss
            return False
        elif current_loss >= self.lowest_loss - self.min_delta:
            # i.e., the current loss is not an improvement in this epoch
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                return True
            else:
                return False
        else:
            # Accept the new lowest loss
            self.lowest_loss = current_loss
            self.epochs_without_improvement = 0
            return False


class ModelTraining:
    """Class for running model training.
    """
    def __init__(
        self,
        model: nn.Module,
        batch_size: int,
        shuffle_batch: bool=False,
        optimizer_class: optim.Optimizer=optim.Adam,
        optimizer_args: dict=None):
        """
        args:
            model: PyTorch model
            optimizer_class: PyTorch optimizer class
        """
        self.model = model
        self.init_model = copy.deepcopy(model)

        self.batch_size = batch_size
        self.shuffle_batch = shuffle_batch
        self.train_loader = None
        self.test_loader = None
        self.validation_loader = None

        self.optimizer_class = optimizer_class
        if optimizer_args is not None:
            self.optimizer = self.optimizer_class(
                self.model.parameters(), **optimizer_args)
        else:
            self.optimizer = self.optimizer_class(self.model.parameters())


    def reset_model(self):
        self.model = copy.deepcopy(self.init_model)
        self.optimizer = self.optimizer_class(self.model.parameters())

    def set_training_data(self, X_train: pd.DataFrame, Y_train: pd.DataFrame):
        assert sorted(X_train.index) == sorted(Y_train.index)
        _, self.train_loader = utils.df_to_datastructs(
            X_train, Y_train, batch_size=self.batch_size,
            shuffle=self.shuffle_batch, index=X_train.index.tolist())

    def set_test_data(self, X_test: pd.DataFrame, Y_test: pd.DataFrame):
        # Shuffle is set to False so that test set IDs can be matched
        assert sorted(X_test.index) == sorted(Y_test.index)
        _, self.test_loader = utils.df_to_datastructs(
            X_test, Y_test, batch_size=self.batch_size,
            shuffle=False, index=X_test.index.tolist())

    def set_validation_data(self, X_valid: pd.DataFrame, Y_valid: pd.DataFrame):
        assert sorted(X_valid.index) == sorted(Y_valid.index)
        _, self.validation_loader = utils.df_to_datastructs(
            X_valid, Y_valid, batch_size=self.batch_size,
            shuffle=False, index=X_valid.index.tolist())
        self.validation = True

    def train(self,
        n_epochs: int,
        criterion: callable,
        test_criterion: callable=None,
        colnames: list=None,
        early_stopping_handler: EarlyStopping=None,
        output_training_preds: bool=False,
        apply_sigmoid: bool=True):
        """Run training over epochs for a Torch module given dataloaders.
        args:
            criterion: criterion function for evaluating predictions
            test_criterion: criterion function for evaluating predictions on test set
            n_epochs: number of epochs to perform training
            colnames: column names for the predictions
            early_stopping_handler: class for performing early stopping evaluation
            output_train_preds: whether or not training predictions should be in output
            apply_sigmoid: whether to apply sigmoid transform to model predictions

        returns: dict with the following:
            'losses': dataframe with losses over epochs
            'preds': dataframe with test set predictions over epochs

            optional:
            'train_preds': dataframe with training set predictions over epochs
        """
        if self.train_loader is None or self.test_loader is None:
            raise AttributeError('Data structures have not been set.')

        if test_criterion is None:
            test_criterion = criterion

        # Train model over epochs and collect results
        losses_over_epochs = []
        train_preds_over_epochs = []
        test_preds_over_epochs = []
        valid_preds_over_epochs = []

        for epoch in range(n_epochs):
            train_losses = []
            test_losses = []
            train_index = []
            train_preds = []
            test_index = []
            test_set_preds = []
            valid_losses = []
            valid_index = []
            valid_preds = []

            self.model.train()
            for ix, (xb, yb) in self.train_loader:
                if torch.is_tensor(ix):
                    ix = ix.data.numpy()
                train_index.append(ix)
                model_output = self.model(xb)
                if apply_sigmoid is True:
                    train_preds.append(torch.sigmoid(model_output))
                else:
                    train_preds.append(model_output)

                loss = criterion(model_output, yb)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.data)

            self.model.eval()
            with torch.no_grad():
                for ix, (xb, yb) in self.test_loader:
                    if torch.is_tensor(ix):
                        ix = ix.data.numpy()
                    test_index.append(ix)
                    model_output = self.model(xb)
                    loss = criterion(model_output, yb)
                    test_losses.append(loss.data)

                    if apply_sigmoid is True:
                        test_set_preds.append(torch.sigmoid(model_output))
                    else:
                        test_set_preds.append(model_output)

                if self.validation_loader is not None:
                    for ix, (xb, yb) in self.validation_loader:
                        if torch.is_tensor(ix):
                            ix = ix.data.numpy()
                        valid_index.append(ix)
                        model_output = self.model(xb)
                        loss = criterion(model_output, yb)
                        valid_losses.append(loss.data)

                        if apply_sigmoid is True:
                            valid_preds.append(torch.sigmoid(model_output))
                        else:
                            valid_preds.append(model_output)

            # Mean training and test losses across batches
            mean_train_loss = np.mean(train_losses)
            mean_test_loss = np.mean(test_losses)
            epoch_results = {
                'train_loss': [mean_train_loss],
                'test_loss': [mean_test_loss]}
            if len(valid_losses) > 0:
                epoch_results['valid_loss'] = [np.mean(valid_losses)]

            current_epoch_df = pd.DataFrame.from_dict(epoch_results)
            current_epoch_df['epoch'] = epoch
            losses_over_epochs.append(current_epoch_df)

            # test_index is a list of tensors, where each list item
            # corresponds to the index of that batch
            complete_test_index = [ix for batch_indices in test_index
            for ix in batch_indices]
            current_epoch_preds = pd.DataFrame(
                torch.cat(test_set_preds, dim=0).data.numpy(),
                columns=colnames, index=complete_test_index)
            current_epoch_preds['epoch'] = epoch
            test_preds_over_epochs.append(current_epoch_preds)

            if output_training_preds is True:
                complete_train_index = [ix for batch_indices in train_index
                for ix in batch_indices]
                current_epoch_train_preds = pd.DataFrame(
                    torch.cat(train_preds, dim=0).data.numpy(),
                    columns=colnames, index=complete_train_index)
                current_epoch_train_preds['epoch'] = epoch
                train_preds_over_epochs.append(current_epoch_train_preds)

            if len(valid_preds) > 0:
                complete_valid_index = [ix for batch_indices in valid_index
                for ix in batch_indices]
                current_epoch_preds = pd.DataFrame(
                    torch.cat(valid_preds, dim=0).data.numpy(),
                    columns=colnames, index=complete_valid_index)
                current_epoch_preds['epoch'] = epoch
                valid_preds_over_epochs.append(current_epoch_preds)

            # Evaluate for early stopping
            if early_stopping_handler is not None:
                early_stop = early_stopping_handler.evaluate_epoch_loss(
                    mean_test_loss)
                if early_stop is True:
                    break

        training_output = {
            'losses': pd.concat(losses_over_epochs),
            'preds': pd.concat(test_preds_over_epochs)}
        if output_training_preds is True:
            training_output['train_preds'] = pd.concat(train_preds_over_epochs)
        if len(valid_preds_over_epochs) > 0:
            training_output['valid_preds'] = pd.concat(valid_preds_over_epochs)

        return training_output


class MixedOutputTraining(ModelTraining):
    def __init__(
        self,
        model: nn.Module,
        reg_cols: list,
        class_cols: list,
        batch_size: int,
        shuffle_batch: bool=False,
        optimizer_class: optim.Optimizer=optim.Adam,
        scaler=StandardScaler()):
        """
        args:
            model: PyTorch model
            reg_cols: column names corresponding to regression in Y
            class_cols: column names corresponding to classification in Y
            optimizer_class: PyTorch optimizer class
        """
        super().__init__(model=model, batch_size=batch_size, shuffle_batch=shuffle_batch,
            optimizer_class=optimizer_class)
        self.reg_cols = reg_cols
        self.class_cols = class_cols

        # Handle the case where there is only one mode in the mixed output
        # training
        if len(self.reg_cols) == 0:
            self.training_mode = 'classification_only'
        elif len(self.class_cols) == 0:
            self.training_mode = 'regression_only'
        else:
            self.training_mode = 'mixed'

        # Scaler to work with transformed regression outputs
        self.scaler = scaler

    def set_training_data(self, X_train: pd.DataFrame, Y_train: pd.DataFrame):
        if len(self.reg_cols) != 0:
            scaled_regression_Y = pd.DataFrame(
                self.scaler.fit_transform(Y_train.loc[:, self.reg_cols]).copy(),
                columns=self.reg_cols,
                index=Y_train.index)
            Y_train = pd.merge(scaled_regression_Y, Y_train[self.class_cols],
                left_index=True, right_index=True)

        assert sorted(X_train.index) == sorted(Y_train.index)
        _, self.train_loader = utils.df_to_datastructs(
            X_train, Y_train, batch_size=self.batch_size, shuffle=self.shuffle_batch,
            index=X_train.index)
        # Find column indices corresponding to classification and regression tasks
        self.train_reg_ix = np.argwhere(Y_train.columns.isin(self.reg_cols)).ravel()
        self.train_class_ix = np.argwhere(Y_train.columns.isin(self.class_cols)).ravel()

    def set_test_data(self, X_test: pd.DataFrame, Y_test: pd.DataFrame):
        if len(self.reg_cols) != 0:
            scaled_regression_Y = pd.DataFrame(
                self.scaler.transform(Y_test.loc[:, self.reg_cols].copy()),
                columns=self.reg_cols,
                index=Y_test.index)
            Y_test = pd.merge(scaled_regression_Y, Y_test[self.class_cols],
                left_index=True, right_index=True)

        assert sorted(X_test.index) == sorted(Y_test.index)

        # Shuffle is set to False so that test set IDs can be matched
        _, self.test_loader = utils.df_to_datastructs(
            X_test, Y_test, batch_size=self.batch_size, shuffle=False,
            index=X_test.index)
        # Find column indices corresponding to classification and regression tasks
        self.test_reg_ix = np.argwhere(Y_test.columns.isin(self.reg_cols)).ravel()
        self.test_class_ix = np.argwhere(Y_test.columns.isin(self.class_cols)).ravel()

    def train(self,
        n_epochs: int,
        reg_criterion: callable,
        class_criterion: callable,
        early_stopping_handler: EarlyStopping=None,
        output_training_preds: bool=False):
        """Run training over epochs for a Torch module given dataloaders.
        args:
            criterion: criterion function for evaluating predictions
            n_epochs: number of epochs to perform training
            early_stopping_handler: class for performing early stopping evaluation
            output_train_preds: whether or not training predictions should be in output

        returns: dict with the following:
            'losses': dataframe with losses over epochs
            'preds': dataframe with test set predictions over epochs

            optional:
            'train_preds': dataframe with training set predictions over epochs
        """
        if self.train_loader is None or self.test_loader is None:
            raise AttributeError('Data structures have not been set.')

        # Train model over epochs and collect results
        losses_over_epochs = []
        train_preds_over_epochs = []
        test_preds_over_epochs = []

        for epoch in range(n_epochs):
            train_losses = []
            test_losses = []
            train_index = []
            train_reg_preds = []
            train_class_preds = []

            test_index = []
            test_set_reg_preds = []
            test_set_class_preds = []

            self.model.train()
            for ix, (xb, yb) in self.train_loader:
                if torch.is_tensor(ix):
                    ix = ix.data.numpy()
                train_index.append(ix)
                model_output = self.model(xb)
                reg_output = model_output['regression_out']
                class_output = model_output['class_out']

                # Calculate losses and progress
                if self.training_mode == 'regression_only':
                    total_loss = reg_criterion(
                        class_output, yb[:, self.train_reg_ix])
                elif self.training_mode == 'classification_only':
                    total_loss = class_criterion(
                        class_output, yb[:, self.train_class_ix])
                else:
                    reg_loss = reg_criterion(
                        reg_output, yb[:, self.train_reg_ix])
                    class_loss = class_criterion(
                        class_output, yb[:, self.train_class_ix])
                    total_loss = reg_loss + class_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                train_losses.append(total_loss.data)

                # Save model outputs
                if self.training_mode != 'classification_only':
                    train_reg_preds.append(
                        self.scaler.inverse_transform(reg_output.data.numpy()))
                if self.training_mode != 'regression_only':
                    train_class_preds.append(torch.sigmoid(class_output))

            self.model.eval()
            with torch.no_grad():
                for ix, (xb, yb) in self.test_loader:
                    if torch.is_tensor(ix):
                        ix = ix.data.numpy()
                    test_index.append(ix)
                    model_output = self.model(xb)
                    reg_output = model_output['regression_out']
                    class_output = model_output['class_out']

                    if self.training_mode == 'regression_only':
                        total_loss = reg_criterion(
                            reg_output, yb[:, self.test_reg_ix])
                    elif self.training_mode == 'classification_only':
                        total_loss = class_criterion(
                            class_output, yb[:, self.test_class_ix])
                    else:
                        reg_loss = reg_criterion(
                            reg_output, yb[:, self.test_reg_ix])
                        class_loss = class_criterion(
                            class_output, yb[:, self.test_class_ix])
                        total_loss = reg_loss + class_loss

                    test_losses.append(total_loss.data)

                    if self.training_mode != 'classification_only':
                        test_set_reg_preds.append(
                            self.scaler.inverse_transform(reg_output.data.numpy()))
                    if self.training_mode != 'regression_only':
                        test_set_class_preds.append(torch.sigmoid(class_output))

            # Mean training and test losses across batches
            mean_train_loss = np.mean(train_losses)
            mean_test_loss = np.mean(test_losses)
            epoch_results = {
                'train_loss': [mean_train_loss],
                'test_loss': [mean_test_loss]}
            current_epoch_df = pd.DataFrame.from_dict(epoch_results)
            current_epoch_df['epoch'] = epoch
            losses_over_epochs.append(current_epoch_df)

            # Handle predictions across batches
            complete_test_index = [
                ix for batch_indices in test_index for ix in batch_indices]
            if len(test_set_reg_preds) != 0:
                current_epoch_reg_preds = pd.DataFrame(
                    np.concatenate(test_set_reg_preds),
                    columns=self.reg_cols,
                    index=complete_test_index)
            if len(test_set_class_preds) != 0:
                current_epoch_class_preds = pd.DataFrame(
                    torch.cat(test_set_class_preds, dim=0).data.numpy(),
                    columns=self.class_cols,
                    index=complete_test_index)

            if self.training_mode == 'classification_only':
                current_epoch_preds = current_epoch_class_preds
            elif self.training_mode == 'regression_only':
                current_epoch_preds = current_epoch_reg_preds
            else:
                current_epoch_preds = pd.merge(
                    current_epoch_reg_preds,
                    current_epoch_class_preds,
                    left_index=True, right_index=True)
            current_epoch_preds['epoch'] = epoch
            test_preds_over_epochs.append(current_epoch_preds)

            if output_training_preds is True:
                complete_train_index = [
                ix for batch_indices in train_index for ix in batch_indices]
                if len(train_reg_preds) != 0:
                    current_epoch_train_reg_preds = pd.DataFrame(
                        np.concatenate(train_reg_preds),
                        columns=self.reg_cols,
                        index=complete_train_index)
                if len(train_class_preds != 0):
                    current_epoch_train_class_preds = pd.DataFrame(
                        torch.cat(train_class_preds, dim=0).data.numpy(),
                        columns=self.class_cols,
                        index=complete_train_index)

                if self.training_mode == 'classification_only':
                    current_epoch_train_preds = current_epoch_train_class_preds
                elif self.training_mode == 'regression_only':
                    current_epoch_train_preds = current_epoch_train_reg_preds
                else:
                    current_epoch_train_preds = pd.merge(
                        current_epoch_train_reg_preds,
                        current_epoch_train_class_preds,
                        left_index=True, right_index=True)

                current_epoch_train_preds['epoch'] = epoch
                train_preds_over_epochs.append(current_epoch_train_preds)

            # Evaluate for early stopping
            if early_stopping_handler is not None:
                early_stop = early_stopping_handler.evaluate_epoch_loss(
                    mean_test_loss)
                if early_stop is True:
                    break

        if output_training_preds is False:
            return {'losses': pd.concat(losses_over_epochs),
                    'preds': pd.concat(test_preds_over_epochs)}
        else:
            return {'losses': pd.concat(losses_over_epochs),
                    'preds': pd.concat(test_preds_over_epochs),
                    'train_preds': pd.concat(train_preds_over_epochs)}


class BottleneckModelTraining(ModelTraining):
    """Class for running model training with bottleneck models.
    In particular, want to keep bottleneck layer outputs for further
    evaluation.
    """
    def __init__(
        self,
        model: nn.Module,
        batch_size: int,
        shuffle_batch: bool=False,
        optimizer_class: optim.Optimizer=optim.Adam):
        """
        args:
            model: PyTorch model
            optimizer_class: PyTorch optimizer class
        """
        super().__init__(
            model=model, batch_size=batch_size, shuffle_batch=shuffle_batch,
            optimizer_class=optimizer_class)

    def train(self,
        n_epochs: int,
        criterion: callable,
        colnames: list=None,
        early_stopping_handler: EarlyStopping=None,
        output_training_preds: bool=False):
        """Run training over epochs for a Torch module given dataloaders.
        args:
            criterion: criterion function for evaluating predictions
            n_epochs: number of epochs to perform training
            colnames: column names for the predictions
            early_stopping_handler: class for performing early stopping evaluation
            output_train_preds: whether or not training predictions should be in output

        returns: dict with the following:
            'losses': dataframe with losses over epochs
            'preds': dataframe with test set predictions over epochs
            'bottleneck': dataframe with bottleneck outputs over epochs

            optional:
            'train_preds': dataframe with training set predictions over epochs
        """
        if self.train_loader is None or self.test_loader is None:
            raise AttributeError('Data structures have not been set.')

        # Train model over epochs and collect results
        losses_over_epochs = []
        train_preds_over_epochs = []
        test_preds_over_epochs = []
        test_bottleneck_over_epochs = []
        valid_preds_over_epochs = []
        valid_bottleneck_over_epochs = []

        for epoch in range(n_epochs):
            train_losses = []
            test_losses = []
            train_index = []
            train_preds = []
            test_index = []
            test_set_preds = []
            test_bottleneck = []
            valid_losses = []
            valid_index = []
            valid_preds = []
            valid_bottleneck = []

            self.model.train()
            for ix, (xb, yb) in self.train_loader:
                if torch.is_tensor(ix):
                    ix = ix.data.numpy()
                train_index.append(ix)
                model_output = self.model(xb)
                train_preds.append(torch.sigmoid(model_output))
                loss = criterion(model_output, yb)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.data)

            self.model.eval()
            with torch.no_grad():
                for ix, (xb, yb) in self.test_loader:
                    if torch.is_tensor(ix):
                        ix = ix.data.numpy()
                    test_index.append(ix)
                    model_output, bottleneck_output = self.model(xb, return_bottleneck=True)
                    loss = criterion(model_output, yb)
                    test_losses.append(loss.data)
                    test_set_preds.append(torch.sigmoid(model_output))
                    test_bottleneck.append(bottleneck_output)

                if self.validation_loader is not None:
                    for ix, (xb, yb) in self.validation_loader:
                        if torch.is_tensor(ix):
                            ix = ix.data.numpy()
                        valid_index.append(ix)
                        model_output, bottleneck_output = self.model(
                            xb, return_bottleneck=True)
                        loss = criterion(model_output, yb)
                        valid_losses.append(loss.data)
                        valid_preds.append(torch.sigmoid(model_output))
                        valid_bottleneck.append(bottleneck_output)

            # Mean training and test losses across batches
            mean_train_loss = np.mean(train_losses)
            mean_test_loss = np.mean(test_losses)
            epoch_results = {
                'train_loss': [mean_train_loss],
                'test_loss': [mean_test_loss]}
            if len(valid_losses) > 0:
                epoch_results['valid_loss'] = [np.mean(valid_losses)]
            current_epoch_df = pd.DataFrame.from_dict(epoch_results)
            current_epoch_df['epoch'] = epoch
            losses_over_epochs.append(current_epoch_df)

            complete_test_index = [
                ix for batch_indices in test_index for ix in batch_indices]
            current_epoch_preds = pd.DataFrame(
                torch.cat(test_set_preds, dim=0).data.numpy(),
                columns=colnames, index=complete_test_index)
            current_epoch_preds['epoch'] = epoch
            test_preds_over_epochs.append(current_epoch_preds)

            test_bottleneck_outputs = torch.cat(test_bottleneck, dim=0).data.numpy()
            n_bottleneck = test_bottleneck_outputs.shape[1]
            current_epoch_test_bottleneck = pd.DataFrame(
                test_bottleneck_outputs,
                columns=['bottleneck_unit_' + str(i) for i in range(n_bottleneck)],
                index=complete_test_index)
            current_epoch_test_bottleneck['epoch'] = epoch
            test_bottleneck_over_epochs.append(current_epoch_test_bottleneck)

            if output_training_preds is True:
                complete_train_index = [
                ix for batch_indices in train_index for ix in batch_indices]
                current_epoch_train_preds = pd.DataFrame(
                    torch.cat(train_preds, dim=0).data.numpy(),
                    columns=colnames, index=complete_train_index)
                current_epoch_train_preds['epoch'] = epoch
                train_preds_over_epochs.append(current_epoch_train_preds)

            if len(valid_preds) > 0:
                complete_valid_index = [ix for batch_indices in valid_index
                for ix in batch_indices]
                current_epoch_preds = pd.DataFrame(
                    torch.cat(valid_preds, dim=0).data.numpy(),
                    columns=colnames, index=complete_valid_index)
                current_epoch_preds['epoch'] = epoch
                valid_preds_over_epochs.append(current_epoch_preds)

                valid_bottleneck_outputs = torch.cat(valid_bottleneck, dim=0).data.numpy()
                n_bottleneck = valid_bottleneck_outputs.shape[1]
                current_epoch_valid_bottleneck = pd.DataFrame(
                    valid_bottleneck_outputs,
                    columns=['bottleneck_unit_' + str(i) for i in range(n_bottleneck)],
                    index=complete_valid_index)
                current_epoch_valid_bottleneck['epoch'] = epoch
                valid_bottleneck_over_epochs.append(current_epoch_valid_bottleneck)

            # Evaluate for early stopping
            if early_stopping_handler is not None:
                early_stop = early_stopping_handler.evaluate_epoch_loss(
                    mean_test_loss)
                if early_stop is True:
                    break

        results_dict = {
            'losses': pd.concat(losses_over_epochs),
            'preds': pd.concat(test_preds_over_epochs),
            'bottleneck': pd.concat(test_bottleneck_over_epochs)
            }

        if output_training_preds is True:
            results_dict['train_preds'] = pd.concat(train_preds_over_epochs)
        if len(valid_preds_over_epochs) > 0:
            results_dict['valid_preds'] = pd.concat(valid_preds_over_epochs)
            results_dict['valid_bottleneck'] = pd.concat(valid_bottleneck_over_epochs)

        return results_dict


class CovariateBottleneckTraining(BottleneckModelTraining):
    def __init__(
        self,
        model: nn.Module,
        batch_size: int,
        covariates: List[str],
        shuffle_batch: bool=False,
        optimizer_class: optim.Optimizer=optim.Adam):
        """
        Training handler for models that incorporate additional covariates in training.
        Covariates are expected to be columns in Y to be extracted.
        args:
            model: PyTorch model
            covariates: covariates to incorporate into model, should be column names in Y
            optimizer_class: PyTorch optimizer class
        """
        super().__init__(
            model=model, batch_size=batch_size, shuffle_batch=shuffle_batch,
            optimizer_class=optimizer_class)
        self.covariates = covariates

    def set_training_data(
        self, X_train: pd.DataFrame, Y_train: pd.DataFrame):
        assert sorted(X_train.index) == sorted(Y_train.index)
        cov_train = Y_train[self.covariates]
        Y_train = Y_train.drop(columns=self.covariates)
        _, self.train_loader = utils.df_to_datastructs(
            X_train, Y_train, cov_train, batch_size=self.batch_size,
            shuffle=self.shuffle_batch, index=X_train.index.tolist())

    def set_test_data(
        self, X_test: pd.DataFrame, Y_test: pd.DataFrame):
        # Shuffle is set to False so that test set IDs can be matched
        assert sorted(X_test.index) == sorted(Y_test.index)
        cov_test = Y_test[self.covariates]
        Y_test = Y_test.drop(columns=self.covariates)
        _, self.test_loader = utils.df_to_datastructs(
            X_test, Y_test, cov_test, batch_size=self.batch_size,
            shuffle=False, index=X_test.index.tolist())

    def train(self,
        n_epochs: int,
        criterion: callable,
        colnames: list=None,
        early_stopping_handler: EarlyStopping=None,
        output_training_preds: bool=False):
        """Run training over epochs for a Torch module given dataloaders.
        args:
            criterion: criterion function for evaluating predictions
            n_epochs: number of epochs to perform training
            colnames: column names for the predictions
            early_stopping_handler: class for performing early stopping evaluation
            output_train_preds: whether or not training predictions should be in output

        returns: dict with the following:
            'losses': dataframe with losses over epochs
            'preds': dataframe with test set predictions over epochs
            'bottleneck': dataframe with bottleneck outputs over epochs

            optional:
            'train_preds': dataframe with training set predictions over epochs
        """
        if self.train_loader is None or self.test_loader is None:
            raise AttributeError('Data structures have not been set.')

        # Train model over epochs and collect results
        losses_over_epochs = []
        train_preds_over_epochs = []
        test_preds_over_epochs = []
        bottleneck_over_epochs = []

        for epoch in range(n_epochs):
            train_losses = []
            test_losses = []
            train_index = []
            train_preds = []
            test_index = []
            test_set_preds = []
            bottleneck = []

            self.model.train()
            for ix, (xb, yb, cb) in self.train_loader:
                if torch.is_tensor(ix):
                    ix = ix.data.numpy()
                train_index.append(ix)
                model_output = self.model(xb, cb)
                train_preds.append(torch.sigmoid(model_output))
                loss = criterion(model_output, yb)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.data)

            self.model.eval()
            with torch.no_grad():
                for ix, (xb, yb, cb) in self.test_loader:
                    if torch.is_tensor(ix):
                        ix = ix.data.numpy()
                    test_index.append(ix)
                    model_output, bottleneck_output = self.model(xb, cb, return_bottleneck=True)
                    loss = criterion(model_output, yb)
                    test_losses.append(loss.data)
                    test_set_preds.append(torch.sigmoid(model_output))
                    bottleneck.append(bottleneck_output)

            # Mean training and test losses across batches
            mean_train_loss = np.mean(train_losses)
            mean_test_loss = np.mean(test_losses)
            epoch_results = {
                'train_loss': [mean_train_loss],
                'test_loss': [mean_test_loss]}
            current_epoch_df = pd.DataFrame.from_dict(epoch_results)
            current_epoch_df['epoch'] = epoch
            losses_over_epochs.append(current_epoch_df)

            complete_test_index = [
                ix for batch_indices in test_index for ix in batch_indices]
            current_epoch_preds = pd.DataFrame(
                torch.cat(test_set_preds, dim=0).data.numpy(),
                columns=colnames, index=complete_test_index)
            current_epoch_preds['epoch'] = epoch
            test_preds_over_epochs.append(current_epoch_preds)

            bottleneck_outputs = torch.cat(bottleneck, dim=0).data.numpy()
            n_bottleneck = bottleneck_outputs.shape[1]
            current_epoch_bottleneck = pd.DataFrame(
                bottleneck_outputs,
                columns=['bottleneck_unit_' + str(i) for i in range(n_bottleneck)],
                index=complete_test_index)
            current_epoch_bottleneck['epoch'] = epoch
            bottleneck_over_epochs.append(current_epoch_bottleneck)

            if output_training_preds is True:
                complete_train_index = [
                ix for batch_indices in train_index for ix in batch_indices]
                current_epoch_train_preds = pd.DataFrame(
                    torch.cat(train_preds, dim=0).data.numpy(),
                    columns=colnames, index=complete_train_index)
                current_epoch_train_preds['epoch'] = epoch
                train_preds_over_epochs.append(current_epoch_train_preds)

            # Evaluate for early stopping
            if early_stopping_handler is not None:
                early_stop = early_stopping_handler.evaluate_epoch_loss(
                    mean_test_loss)
                if early_stop is True:
                    break

        results_dict = {
            'losses': pd.concat(losses_over_epochs),
            'preds': pd.concat(test_preds_over_epochs),
            'bottleneck': pd.concat(bottleneck_over_epochs)
            }

        if output_training_preds is False:
            return results_dict
        else:
            results_dict['train_preds'] = pd.concat(train_preds_over_epochs)
            return results_dict
