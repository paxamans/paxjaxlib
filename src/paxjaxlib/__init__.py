__version__ = '0.0.4'

from .activations import relu, linear, sigmoid, tanh, softmax
from .layers import Dense, Conv2D, Flatten, Dropout # Added Dropout
from .models import NeuralNetwork
from .training import Trainer
from .losses import mse_loss, binary_crossentropy, categorical_crossentropy
from .optimizers import SGD, Adam, AdaMax, RMSprop, Momentum, Adafactor, AdaGrad, Adadelta
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
