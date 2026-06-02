"""
paxjaxlib — A simple and functional neural network library built on JAX.

Quickstart::

    from paxjaxlib import Dense, NeuralNetwork, Trainer
    import jax, optax

    key = jax.random.PRNGKey(0)
    model = NeuralNetwork([Dense(10, 64, key), jax.nn.relu, Dense(64, 1, key)])
    trainer = Trainer(model, optimizer=optax.adam(1e-3))
    history = trainer.train(X, y, epochs=10)
"""

__version__ = "0.2.0"

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
from .activations import (
    elu,
    gelu,
    leaky_relu,
    linear,
    mish,
    relu,
    sigmoid,
    silu,
    softmax,
    tanh,
)
from .layers import (
    AvgPooling2D,
    BatchNorm,
    Conv2D,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    GlobalAvgPooling2D,
    LayerNorm,
    MaxPooling2D,
    MultiHeadAttention,
)
from .losses import (
    binary_crossentropy,
    categorical_crossentropy,
    cosine_similarity_loss,
    hinge_loss,
    huber_loss,
    kl_divergence,
    mse_loss,
)
from .models import NeuralNetwork
from .training import Trainer

# Conditionally expose the PyTorch bridge when torch is installed.
try:
    from . import torch_bridge  # noqa: F401
except ImportError:
    pass

__all__ = [
    # Layers
    "AvgPooling2D",
    "BatchNorm",
    "Conv2D",
    "Dense",
    "Dropout",
    "Embedding",
    "Flatten",
    "GlobalAvgPooling2D",
    "LayerNorm",
    "MaxPooling2D",
    "MultiHeadAttention",
    # Model
    "NeuralNetwork",
    # Training
    "Trainer",
    # Losses
    "mse_loss",
    "binary_crossentropy",
    "categorical_crossentropy",
    "cosine_similarity_loss",
    "hinge_loss",
    "huber_loss",
    "kl_divergence",
    # Activations
    "elu",
    "gelu",
    "leaky_relu",
    "linear",
    "mish",
    "relu",
    "sigmoid",
    "silu",
    "softmax",
    "tanh",
    # Sub-modules
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
