"""Bottlenecked deep learning models."""
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim


from biobank_project.deep_mtl.models import base, ensemble
from biobank_project.deep_mtl.models.bottleneck import EnsembleNetwork


class EnsembleSkipNetwork(nn.Module):
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
        self.bottleneck_integration = nn.Sequential(
            nn.Linear(n_tasks + n_bottleneck, n_tasks),
            nn.ReLU(),
            nn.Linear(n_tasks, n_tasks)
        )

    def forward(self, xb, return_bottleneck=False):
        xb = self.layers(xb)
        bottleneck = self.bottleneck(xb)
        output = self.out(bottleneck)
        output = self.bottleneck_integration(torch.cat([output, bottleneck], axis=1))
        if return_bottleneck is True:
            return output, bottleneck
        else:
            return output

# TODO: Implement a model that has bottleneck injection directly in each model?
class MultiModelInject(nn.Module):
    """Single layer consisting of multiple models.
    Each model receives combined input and model outputs are concatenated together.
    """
    def __init__(self, n_combined_input, n_models, n_hidden=150, n_outputs=150):
        """
        args:
            n_combined_input: total number of combined input.
                (e.g., {N_OUTPUTS} * {N_TASKS} from a previous MultiModelHidden module)
            n_models: number of independent models
            n_hidden: number of hidden units
            n_outputs: number of outputs for each individual model
        """
        super().__init__()
        self.models = nn.ModuleDict(
            {'model_' + str(i): nn.Sequential(
                nn.Linear(n_combined_input, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_outputs))
            for i in range(n_models)})

    def forward(self, xb):
        individual_model_outputs = [model(xb) for model in self.models.values()]
        return torch.cat(individual_model_outputs, dim=1)


# Modified ensemble network that uses simple transform from
# bottleneck
class EnsembleDirect(EnsembleNetwork):
    """Ensemble Network that uses bottleneck layer directly for predictions
    Ensemble network where each layer consists of independent models for each task.
    Models in each layer receive concatenated inputs from all other models of the previous layer.
    """
    def __init__(self, n_features, n_tasks, n_hidden: int=150, n_bottleneck: int=10, transform='linear'):
        super().__init__(n_features=n_features, n_tasks=n_tasks, n_hidden=n_hidden, n_bottleneck=n_bottleneck)

        # Final layer has one output per model, corresponding to all tasks.
        self.out = nn.Sequential(
            MultiModelSimple(
                n_combined_input=n_bottleneck, n_models=n_tasks, n_outputs=1,
                transform=transform))

    def forward(self, xb, return_bottleneck=False):
        xb = self.layers(xb)
        bottleneck = self.bottleneck(xb)
        output = self.out(bottleneck)
        if return_bottleneck is True:
            return output, bottleneck
        else:
            return output

class MultiModelSimple(nn.Module):
    """Modification of Ensemble Network MultiModel No Hidden Units
    Each model receives combined input and model outputs are concatenated together.
    """
    def __init__(self, n_combined_input, n_models, n_outputs=150, transform='nonlinear'):
        """
        args:
            n_combined_input: total number of combined input.
                (e.g., {N_OUTPUTS} * {N_TASKS} from a previous MultiModelHidden module)
            n_models: number of independent models
            n_outputs: number of outputs for each individual model
            transform: 'linear' or 'nonlinear', whether to build in nonlinear transform
        """
        super().__init__()
        ALLOWED_TRANSFORM_ARGS = ['linear', 'nonlinear']
        if transform not in ALLOWED_TRANSFORM_ARGS:
            raise ValueError(
                f'Unrecognized argument for transform, must be one of: {ALLOWED_TRANSFORM_ARGS}')
        if transform == 'nonlinear':
            self.models = nn.ModuleDict(
                {'model_' + str(i): nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(n_combined_input, n_outputs))
                for i in range(n_models)})
        else:
            self.models = nn.ModuleDict(
                {'model_' + str(i): nn.Sequential(
                    nn.Linear(n_combined_input, n_outputs))
                for i in range(n_models)})

    def forward(self, xb):
        individual_model_outputs = [model(xb) for model in self.models.values()]
        return torch.cat(individual_model_outputs, dim=1)


class CovariateEnsemble(nn.Module):
    """
    Class for defining augmented bottleneck models, where covariate
    information is appended in the bottleneck layer.
    """
    def __init__(self, n_features, n_tasks, n_hidden: int=150, n_bottleneck: int=10, n_covariates: int=1):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(n_hidden * n_tasks, n_bottleneck),
            nn.BatchNorm1d(n_bottleneck),
            nn.ReLU()
        )

        self.layers = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            ensemble.MultiModelHidden(n_combined_input=n_hidden, n_models=n_tasks, n_hidden=n_hidden, n_outputs=n_hidden),
            ensemble.MultiModelHidden(n_combined_input=n_hidden * n_tasks, n_models=n_tasks, n_hidden=n_hidden, n_outputs=n_hidden),
        )
        # Final layer has one output per model, corresponding to all tasks.
        self.out = nn.Sequential(
            ensemble.MultiModelHidden(
                n_combined_input=n_bottleneck + n_covariates, n_models=n_tasks, n_hidden=n_hidden, n_outputs=1)
        )

    def forward(self, xb, cb, return_bottleneck=False):
        """args
        where cb is the covariate in the batch
        """
        xb = self.layers(xb)
        bottleneck = self.bottleneck(xb)
        if len(cb.shape) == 1:
            cb = cb.unsqueeze(1)
        bottleneck = torch.cat((bottleneck, cb), dim=1)
        output = self.out(bottleneck)
        if return_bottleneck is True:
            return output, bottleneck
        else:
            return output


class AdditiveCovariate(nn.Module):
    """
    Class for defining augmented bottleneck models, where covariate
    information is appended in the bottleneck layer.
    """
    def __init__(self, n_features, n_tasks, n_hidden: int=150, n_bottleneck: int=10, n_covariates: int=1):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(n_hidden * n_tasks, n_bottleneck),
            nn.BatchNorm1d(n_bottleneck)
        )

        self.adjustment_layer = nn.Sequential(
            nn.Linear(n_covariates, n_bottleneck)
        )

        self.layers = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            ensemble.MultiModelHidden(n_combined_input=n_hidden, n_models=n_tasks, n_hidden=n_hidden, n_outputs=n_hidden),
            ensemble.MultiModelHidden(n_combined_input=n_hidden * n_tasks, n_models=n_tasks, n_hidden=n_hidden, n_outputs=n_hidden),
        )

        # Final layer as multioutputs from bottleneck layer
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(n_bottleneck, n_tasks)
        )

    def forward(self, xb, cb, return_bottleneck=False):
        """args
        where cb is the covariate in the batch
        """
        xb = self.layers(xb)
        bottleneck = self.bottleneck(xb)

        # Use covariates to learn an adjustment factor
        if len(cb.shape) == 1:
            cb = cb.unsqueeze(1)
        adjust_factor = self.adjustment_layer(cb)

        # Adjust bottleneck based on covariates
        adj_bottleneck = bottleneck + adjust_factor * bottleneck
        output = self.out(adj_bottleneck)
        if return_bottleneck is True:
            return output, bottleneck
        else:
            return output


