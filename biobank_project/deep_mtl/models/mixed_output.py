"""Mixed Output Models."""
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim


class ThreeLayerMultiOutput(nn.Module):
    """Multi-Output architecture.
    """
    def __init__(self,
    n_features: int,
    n_class_outputs: int,
    n_reg_outputs: int,
    hidden_layer_spec: dict=None):
        """args:
            n_features: number of input features for the first layer
            n_class_outputs: number of classification outputs
            n_reg_outputs: number of regression outputs
            hidden_layer_spec: dict of ints with the following keys:
                ['hidden_1', 'hidden_2', 'hidden_3'] specifying number of units
                for each hidden layer
        """
        super().__init__()

        if hidden_layer_spec is None:
            hidden_layer_spec = {
                'hidden_1': 100,
                'hidden_2': 100,
                'hidden_3': 100,
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
        )

        self.reg_output = nn.Linear(
            n_hidden_layer_3, n_reg_outputs)
        self.class_output = nn.Linear(
            n_hidden_layer_3, n_class_outputs)

    def forward(self, xb):
        hidden_output = self.layers(xb)
        return {
            'regression_out': self.reg_output(hidden_output),
            'class_out': self.class_output(hidden_output)
        }


# Single task-like architecture that trains independent models
class ThreeLayerSingleTask(nn.Module):
    """Collection of single task models.
    """
    def __init__(self,
    n_features: int,
    n_class_outputs: int,
    n_reg_outputs: int,
    hidden_layer_spec: dict=None):
        """args:
            n_features: number of input features for the first layer
            n_class_outputs: number of classification outputs
            n_reg_outputs: number of regression outputs
            hidden_layer_spec: dict of ints with the following keys:
                ['hidden_1', 'hidden_2', 'hidden_3'] specifying number of units
                for each hidden layer
        """
        super().__init__()

        if hidden_layer_spec is None:
            hidden_layer_spec = {
                'hidden_1': 100,
                'hidden_2': 100,
                'hidden_3': 100,
            }

        self.regression_models = nn.ModuleDict({
            'reg_' + str(i): self.init_base_model(
                n_features, hidden_layer_spec)
            for i in range(n_reg_outputs)
        })
        self.classification_models = nn.ModuleDict({
            'class_' + str(i): self.init_base_model(
                n_features, hidden_layer_spec)
            for i in range(n_class_outputs)
        })

    def init_base_model(self, n_features, hidden_layer_spec):
        n_hidden_layer_1 = hidden_layer_spec['hidden_1']
        n_hidden_layer_2 = hidden_layer_spec['hidden_2']
        n_hidden_layer_3 = hidden_layer_spec['hidden_3']

        return nn.Sequential(
            nn.Linear(n_features, n_hidden_layer_1),
            nn.BatchNorm1d(n_hidden_layer_1),
            nn.ReLU(),
            nn.Linear(n_hidden_layer_1, n_hidden_layer_2),
            nn.BatchNorm1d(n_hidden_layer_2),
            nn.ReLU(),
            nn.Linear(n_hidden_layer_2, n_hidden_layer_3),
            nn.BatchNorm1d(n_hidden_layer_3),
            nn.ReLU(),
            nn.Linear(n_hidden_layer_3, 1)
        )

    def forward(self, xb):
        if len(self.regression_models) != 0:
            reg_outputs = torch.cat([
                self.regression_models[k](xb)
                for k in self.regression_models.keys()], dim=1)
        else:
            reg_outputs = None

        if len(self.classification_models) != 0:
            class_outputs = torch.cat([
                self.classification_models[k](xb)
                for k in self.classification_models.keys()], dim=1)
        else:
            class_outputs = None

        return {
            'regression_out': reg_outputs,
            'class_out': class_outputs
        }


class MultiModelHidden(nn.Module):
    """Single layer consisting of multiple models.
    Each model receives combined input and model outputs are concatenated together.
    """
    def __init__(self,
    n_combined_input,
    n_models,
    n_hidden=150,
    n_outputs=150):
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


class MultiModelMixedOutput(nn.Module):
    """Single layer consisting of multiple models.
    Each model receives combined input and model outputs are concatenated together.
    """
    def __init__(self,
    n_combined_input,
    n_reg_models: int,
    n_class_models: int,
    n_hidden=150,
    n_outputs=1):
        """
        args:
            n_combined_input: total number of combined input.
                (e.g., {N_OUTPUTS} * {N_TASKS} from a previous MultiModelHidden module)
            n_models: number of independent models
            n_hidden: number of hidden units
            n_outputs: number of outputs for each individual model
        """
        super().__init__()
        self.reg_models = nn.ModuleDict(
            {'model_' + str(i): nn.Sequential(
                nn.Linear(n_combined_input, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_outputs))
            for i in range(n_reg_models)})

        self.class_models = nn.ModuleDict(
            {'model_' + str(i): nn.Sequential(
                nn.Linear(n_combined_input, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_outputs))
            for i in range(n_class_models)})

    def forward(self, xb):
        regression_out = None
        class_out = None
        if len(self.reg_models) > 0:
            reg_model_outputs = [model(xb) for model in self.reg_models.values()]
            regression_out = torch.cat(reg_model_outputs, dim=1)
        if len(self.class_models) > 0:
            class_model_outputs = [model(xb) for model in self.class_models.values()]
            class_out = torch.cat(class_model_outputs, dim=1)
        return {
            'regression_out': regression_out,
            'class_out': class_out
        }


class EnsembleNetwork(nn.Module):
    """Ensemble network where each layer consists of independent models for each task.
    Models in each layer receive concatenated inputs from all other models of the previous layer.
    """
    def __init__(self, n_features, n_reg_tasks, n_class_tasks):
        super().__init__()
        n_total_tasks = n_reg_tasks + n_class_tasks
        self.layers = nn.Sequential(
            nn.Linear(n_features, 150),
            MultiModelHidden(
                n_combined_input=150, n_models=n_total_tasks, n_hidden=150, n_outputs=150),
            MultiModelHidden(
                n_combined_input=150 * n_total_tasks, n_models=n_total_tasks,
                n_hidden=150, n_outputs=150),
            # Final layer has one output per model, corresponding to all tasks.
            MultiModelMixedOutput(
                n_combined_input=150*n_total_tasks,
                n_reg_models=n_reg_tasks,
                n_class_models=n_class_tasks,
                n_hidden=150,
                n_outputs=1)
        )

    def forward(self, xb):
        return self.layers(xb)
