"""Bottlenecked deep learning models."""
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim


from biobank_project.deep_mtl.models import base, ensemble


class ThreeLayerBottleneck(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        n_hidden: int=150,
        n_bottleneck: int=10,
        hidden_layer_spec: dict=None):
        """args:
            n_features: number of input features for the first layer
            n_outputs: total outputs (tasks)
            n_bottleneck: number of units for the bottleneck layer
            hidden_layer_spec: dict of ints with the following keys:
                ['hidden_1', 'hidden_2', 'hidden_3'] specifying number of units
                for each hidden layer
        """
        super().__init__()

        if hidden_layer_spec is None:
            hidden_layer_spec = {
                'hidden_1': n_hidden,
                'hidden_2': n_hidden,
                'hidden_3': n_hidden,
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
            nn.Linear(n_hidden_layer_3, n_hidden_layer_3)
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(n_hidden_layer_3, n_bottleneck),
            nn.BatchNorm1d(n_bottleneck),
            nn.ReLU()
        )
        self.out = nn.Linear(n_bottleneck, n_outputs)

    def forward(self, xb, return_bottleneck=False):
        xb = self.layers(xb)
        bottleneck = self.bottleneck(xb)
        output = self.out(bottleneck)
        if return_bottleneck is True:
            return output, bottleneck
        else:
            return output


class EnsembleNetwork(nn.Module):
    """Ensemble network where each layer consists of independent models for each task.
    Models in each layer receive concatenated inputs from all other models of the previous layer.
    """
    def __init__(self, n_features, n_tasks, n_hidden: int=150, n_bottleneck: int=10):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(n_hidden * n_tasks, n_bottleneck),
            nn.BatchNorm1d(n_bottleneck),
            nn.ReLU()
        )

        self.layers = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.Linear(n_hidden, n_hidden * n_tasks),
            nn.ReLU(),
            nn.Linear(n_hidden * n_tasks, n_hidden * n_tasks),
            nn.ReLU()
        )

        # Final layer has one output per model, corresponding to all tasks.
        self.out = nn.Sequential(
            ensemble.MultiModelHidden(n_combined_input=n_bottleneck, n_models=n_tasks, n_hidden=n_hidden, n_outputs=1)
        )

    def forward(self, xb, return_bottleneck=False):
        xb = self.layers(xb)
        bottleneck = self.bottleneck(xb)
        output = self.out(bottleneck)
        if return_bottleneck is True:
            return output, bottleneck
        else:
            return output


class ParallelEnsembleNetwork(nn.Module):
    def __init__(self, n_features, n_tasks, n_hidden: int=150, n_bottleneck: int=10):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(n_hidden * n_tasks, n_bottleneck),
            nn.BatchNorm1d(n_bottleneck),
            nn.ReLU()
        )
        self.layers = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            ensemble.SameLayerConnectedFirstInput(
                prev_layer_input=n_hidden, n_models=n_tasks,
                n_hidden=n_hidden, n_outputs=n_hidden),
            ensemble.SameLayerConnected(
                prev_layer_input=n_hidden, n_models=n_tasks,
                n_hidden=n_hidden, n_outputs=n_hidden),
        )
        self.out = nn.Sequential(
            ensemble.SameLayerConnectedFirstInput(
                prev_layer_input=n_bottleneck, n_models=n_tasks,
                n_hidden=n_hidden, n_outputs=1)
        )

    def forward(self, xb, return_bottleneck=False):
        layer_output_dict = self.layers(xb)
        layer_outputs = [v for v in layer_output_dict.values()]
        bottleneck = self.bottleneck(torch.cat(layer_outputs, dim=1))
        decoder_output_dict = self.out(bottleneck)
        task_decoder_outputs = [v for v in decoder_output_dict.values()]
        task_decoder_outputs = torch.cat(task_decoder_outputs, dim=1)
        if return_bottleneck is True:
            return task_decoder_outputs, bottleneck
        else:
            return task_decoder_outputs
