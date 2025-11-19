__version__ = "0.2.0"

from .activations import linear, relu, sigmoid, softmax, tanh
from .layers import Conv2D, Dense, Dropout, Flatten  # Added Dropout
from .losses import binary_crossentropy, categorical_crossentropy, mse_loss
from .models import NeuralNetwork
from .training import Trainer

# from .schedules import exponential_decay, step_decay # later

__all__ = [
    # Layers
    "Conv2D",
    "Dense",
    "Dropout",  # Added Dropout
    "Flatten",
    # Model
    "NeuralNetwork",
    # Training
    "Trainer",
    # Losses
    "mse_loss",
    "binary_crossentropy",
    "categorical_crossentropy",
    # Activations
    "relu",
    "linear",
    "sigmoid",
    "tanh",
    "softmax",
    # Schedules later
    # 'exponential_decay',
    # 'step_decay',
]

from . import (
    activations,
    core,
    initializers,
    layers,
    losses,
    metrics,
    models,
    regularizers,
    schedules,
    training,
)

__all__ += [
    "activations",
    "core",
    "initializers",
    "layers",
    "losses",
    "metrics",
    "models",
    "regularizers",
    "schedules",
    "training",
]
