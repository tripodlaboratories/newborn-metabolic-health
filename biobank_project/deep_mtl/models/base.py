"""Baseline deep learning models."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Logistic(nn.Module):
    """Define NN architecture that recreates logistic regression.
    """
    def __init__(self, n_features, n_outputs):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_features, n_outputs)
        )

    def forward(self, xb):
        return self.linear(xb)


class SingleHidden(nn.Module):
    """Neural network architecture with a single hidden layer
    """
    def __init__(self, n_features, n_outputs, n_hidden=150):
        super().__init__()
        # Define NN architecture with one variable hidden layer
        self.layers = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_outputs)
        )

    def forward(self, xb):
        return self.layers(xb)


class ThreeLayerMultiOutput(nn.Module):
    """Multi-Output architecture.
    """
    def __init__(self, n_features: int, n_outputs: int, hidden_layer_spec: dict=None):
        """args:
            n_features: number of input features for the first layer
            n_outputs: total outputs (tasks)
            hidden_layer_spec: dict of ints with the following keys:
                ['hidden_1', 'hidden_2', 'hidden_3'] specifying number of units
                for each hidden layer
        """
        super().__init__()

        if hidden_layer_spec is None:
            hidden_layer_spec = {
                'hidden_1': 150,
                'hidden_2': 150,
                'hidden_3': 150,
            }

        n_hidden_layer_1 = hidden_layer_spec['hidden_1']
        n_hidden_layer_2 = hidden_layer_spec['hidden_2']
        n_hidden_layer_3 = hidden_layer_spec['hidden_3']

        self.layers = nn.Sequential(
            nn.Linear(n_features, n_hidden_layer_1),
            nn.BatchNorm1d(n_hidden_layer_1),
            nn.ReLU(),
            nn.Linear(n_hidden_layer_1, n_hidden_layer_2),
            nn.BatchNorm1d(n_hidden_layer_2),
            nn.ReLU(),
            nn.Linear(n_hidden_layer_2, n_hidden_layer_3),
            nn.BatchNorm1d(n_hidden_layer_3),
            nn.ReLU(),
            nn.Linear(n_hidden_layer_3, n_outputs)
        )

    def forward(self, xb):
        return self.layers(xb)


class TwoLayerMultiOutput(nn.Module):
    """Multi-Output architecture.
    """
    def __init__(self, n_features: int, n_outputs: int, hidden_layer_spec: dict=None):
        """args:
            n_features: number of input features for the first layer
            n_outputs: total outputs (tasks)
            hidden_layer_spec: dict of ints with the following keys:
                ['hidden_1', 'hidden_2', 'hidden_3'] specifying number of units
                for each hidden layer
        """
        super().__init__()

        if hidden_layer_spec is None:
            hidden_layer_spec = {
                'hidden_1': 100,
                'hidden_2': 100
            }

        n_hidden_layer_1 = hidden_layer_spec['hidden_1']
        n_hidden_layer_2 = hidden_layer_spec['hidden_2']

        self.layers = nn.Sequential(
            nn.Linear(n_features, n_hidden_layer_1),
            nn.BatchNorm1d(n_hidden_layer_1),
            nn.ReLU(),
            nn.Linear(n_hidden_layer_1, n_hidden_layer_2),
            nn.BatchNorm1d(n_hidden_layer_2),
            nn.ReLU(),
            nn.Linear(n_hidden_layer_2, n_outputs)
        )

    def forward(self, xb):
        return self.layers(xb)


class DeepMultiOutput(nn.Module):
    """Multi-Output architecture.
    """
    def __init__(self, n_features: int,
                 n_outputs: int,
                 hidden_layer_spec: dict=None):
        """args:
            n_features: number of input features for the first layer
            n_outputs: total outputs (tasks)
            hidden_layer_spec: dict of ints with the following keys:
                ['hidden_0', 'hidden_1', 'hidden_2'] specifying number of units
                for each hidden layer
        """
        super().__init__()

        if hidden_layer_spec is None:
            n_hidden_layers = 10
            hidden_layer_spec = {
                'hidden_' + str(i): 100 for i in range(n_hidden_layers)}

        self.first_layer = nn.Sequential(
            nn.Linear(n_features, hidden_layer_spec['hidden_1']),
            nn.BatchNorm1d(hidden_layer_spec['hidden_1']),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList()
        for i, _ in enumerate(hidden_layer_spec.keys()):
            if i == len(hidden_layer_spec.keys()):
                continue
            else:
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_layer_spec['hidden_' + str(i)],
                                  hidden_layer_spec['hidden_' + str(i)]),
                        nn.BatchNorm1d(hidden_layer_spec['hidden_' + str(i)]),
                        nn.ReLU()
                    ))

        self.final_layer = nn.Sequential(
            nn.Linear(
                hidden_layer_spec['hidden_' + str(len(hidden_layer_spec)-1)],
                n_outputs)
        )

    def forward(self, xb):
        xb = self.first_layer(xb)
        for layer in self.layers:
            xb = layer(xb)
        return self.final_layer(xb)

