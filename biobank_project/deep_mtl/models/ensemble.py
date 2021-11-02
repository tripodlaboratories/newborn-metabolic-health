"""Ensemble deep learning models."""
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim


class MultiModelHidden(nn.Module):
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


class EnsembleNetwork(nn.Module):
    """Ensemble network where each layer consists of independent models for each task.
    Models in each layer receive concatenated inputs from all other models of the previous layer.
    """
    def __init__(self, n_features, n_tasks, n_hidden: int=150, n_output_hidden: int=150):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            MultiModelHidden(n_combined_input=n_hidden, n_models=n_tasks, n_hidden=n_hidden, n_outputs=n_hidden),
            MultiModelHidden(n_combined_input=n_hidden * n_tasks, n_models=n_tasks, n_hidden=n_hidden, n_outputs=n_hidden),
            # Final layer has one output per model, corresponding to all tasks.
            MultiModelHidden(
                n_combined_input=n_hidden * n_tasks, n_models=n_tasks,
                n_hidden=n_output_hidden, n_outputs=1)
        )

    def forward(self, xb):
        return self.layers(xb)


class SameLayerConnected(nn.Module):
    def __init__(self, prev_layer_input: int, n_models:int, n_hidden: int=300, n_outputs: int=300):
        super().__init__()
        self.individual_models = nn.ModuleDict(
            {'model_' + str(i): nn.Sequential(
                nn.Linear(prev_layer_input, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_outputs))
             for i in range(n_models)})

        self.shared_models = nn.ModuleDict(
            {'model_' + str(i): nn.Sequential(
                nn.Linear(prev_layer_input + (n_outputs * (n_models - 1)), n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_outputs))
             for i in range(n_models)})

    def forward(self, individual_model_inputs: dict) -> dict:
        individual_model_outputs = {
            name: model(individual_model_inputs[name]) for name, model in self.individual_models.items()}
        shared_model_outputs = {k: None for k in self.shared_models.keys()}

        for current_name, current_model in self.shared_models.items():
            model_specific_input = individual_model_inputs[current_name]
            shared_layer_input = [
                individual_model_outputs[model_name] for model_name in individual_model_outputs.keys()
                if model_name != current_name]
            shared_layer_input = torch.cat(shared_layer_input, dim=1)
            shared_model_outputs[current_name] = current_model(
                torch.cat((model_specific_input, shared_layer_input), dim=1))
        return shared_model_outputs


class SameLayerConnectedFirstInput(SameLayerConnected):
    def __init__(self, prev_layer_input: int, n_models:int, n_hidden: int=300, n_outputs: int=300):
        super().__init__(
            prev_layer_input=prev_layer_input, n_models=n_models, n_hidden=n_hidden, n_outputs=n_outputs)

    def forward(self, xb: torch.Tensor) -> dict:
        individual_model_outputs = {name: model(xb) for name, model in self.individual_models.items()}
        shared_model_outputs = {k: None for k in self.shared_models.keys()}

        for current_name, current_model in self.shared_models.items():
            shared_layer_input = [individual_model_outputs[model_name]
                                  for model_name in individual_model_outputs.keys()
                                  if model_name != current_name]
            shared_layer_input = torch.cat(shared_layer_input, dim=1)
            shared_model_outputs[current_name] = current_model(torch.cat((xb, shared_layer_input), dim=1))

        return shared_model_outputs


class ParallelEnsembleNetwork(nn.Module):
    def __init__(self, n_features, n_tasks, n_hidden: int=150):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            SameLayerConnectedFirstInput(prev_layer_input=n_hidden, n_models=n_tasks, n_hidden=n_hidden, n_outputs=n_hidden),
            SameLayerConnected(prev_layer_input=n_hidden, n_models=n_tasks, n_hidden=n_hidden, n_outputs=n_hidden),
            SameLayerConnected(prev_layer_input=n_hidden, n_models=n_tasks, n_hidden=n_hidden, n_outputs=1)
        )

    def forward(self, xb):
        output_dict = self.layers(xb)
        total_outputs = [v for v in output_dict.values()]
        return torch.cat(total_outputs, dim=1)
