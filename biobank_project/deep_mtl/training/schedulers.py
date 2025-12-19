"""Factory functions for learning rate schedulers, to be used with the ModelTraining Handler."""
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, OneCycleLR

def create_reduce_lr_scheduler(optimizer, config):
    """Creates a ReduceLROnPlateau scheduler."""
    return ReduceLROnPlateau(
        optimizer,
        mode=config.get("mode", "min"),  # Default to 'min' if not specified
        factor=config.get("factor", 0.1),
        patience=config.get("patience", 3),
        verbose=config.get("verbose", True),
    )

def create_step_lr_scheduler(optimizer, config):
    """Creates a StepLR scheduler."""
    return StepLR(
        optimizer,
        step_size=config["step_size"],  # Required parameter
        gamma=config.get("gamma", 0.1),
    )

def create_cosine_annealing_scheduler(optimizer, config):
    """Creates a CosineAnnealingLR scheduler."""
    return CosineAnnealingLR(
        optimizer,
        T_max=config["T_max"],  # Required: total number of epochs
    )

def create_one_cycle_lr_scheduler(optimizer, config, total_steps):
    """Creates a OneCycleLR scheduler."""
    return OneCycleLR(
        optimizer,
        max_lr = config['max_lr'],
        total_steps = total_steps,
        epochs = config.get('epochs'), #Required, but can be None if total_steps is specified.
        steps_per_epoch = config.get('steps_per_epoch'), #Required, but can be None if total_steps is specified
        pct_start = config.get('pct_start', 0.3),
        anneal_strategy = config.get('anneal_strategy', 'cos'),
        div_factor = config.get('div_factor', 25.0),
        final_div_factor = config.get('final_div_factor', 1e4)
    )

def get_scheduler_creator(scheduler_name):
    """
    Returns the appropriate scheduler creator function based on the name.
    """
    if scheduler_name == "ReduceLROnPlateau":
        return create_reduce_lr_scheduler
    elif scheduler_name == "StepLR":
        return create_step_lr_scheduler
    elif scheduler_name == "CosineAnnealingLR":
        return create_cosine_annealing_scheduler
    elif scheduler_name == "OneCycleLR":
        return create_one_cycle_lr_scheduler
    else:
        raise ValueError(f"Unknown scheduler name: {scheduler_name}")