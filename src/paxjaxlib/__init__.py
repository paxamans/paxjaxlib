__version__ = '0.0.4'

from .activations import linear, relu, sigmoid, softmax, tanh
from .layers import Conv2D, Dense, Dropout, Flatten  # Added Dropout
from .losses import binary_crossentropy, categorical_crossentropy, mse_loss
from .models import NeuralNetwork
from .optimizers import (
    SGD,
    Adadelta,
    Adafactor,
    AdaGrad,
    Adam,
    AdaMax,
    Momentum,
    RMSprop,
)
from .training import Trainer

# from .schedules import exponential_decay, step_decay # later

__all__ = [
    # Layers
    'Conv2D',
    'Dense',
    'Dropout', # Added Dropout
    'Flatten',
    # Model
    'NeuralNetwork',
    # Training
    'Trainer',
    # Losses
    'mse_loss',
    'binary_crossentropy',
    'categorical_crossentropy',
    # Activations
    'relu',
    'linear',
    'sigmoid',
    'tanh',
    'softmax',
    # Optimizers
    'SGD',
    'Adam',
    'AdaMax',
    'RMSprop',
    'Momentum',
    'Adafactor',
    'AdaGrad',
    'Adadelta'
    # Schedules later
    # 'exponential_decay',
    # 'step_decay',
]
__all__.sort()
