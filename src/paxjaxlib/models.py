from typing import Any, List, Optional

import jax.numpy as jnp
from jax import random

from .core import Module
from .layers import BatchNorm, Conv2D, Dense, Dropout, LayerNorm


class NeuralNetwork(Module):
    def __init__(self, layers: List[Module]):
        super().__init__()
        self.layers = layers
        self._built = False

    @property
    def params(self):
        """Collect parameters from all layers that have them."""
        params = []
        for layer in self.layers:
            if hasattr(layer, "params"):
                params.append(layer.params)
        return params

    def build(self, input_shape, key):
        if self._built:
            return
        for layer in self.layers:
            if isinstance(layer, (Dense, Conv2D, BatchNorm, LayerNorm)):
                layer.build(input_shape, key)
            input_shape = layer(jnp.ones(input_shape)).shape
        self._built = True

    def __call__(
        self,
        X: jnp.ndarray,
        key: Optional[Any] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        """
        Forward pass through the network.
        """
        # Clear losses before forward pass
        self.clear_losses()

        current_input = X

        # If key is provided, split it for layers that need it
        iter_key = key

        for layer in self.layers:
            # Check if layer needs key/training args
            # We can check signature or just try/except, or check type.
            # Checking type is safer for our known layers.

            if isinstance(layer, Dropout):
                if training and iter_key is not None:
                    iter_key, subkey = random.split(iter_key)
                    current_input = layer(current_input, key=subkey, training=training)
                else:
                    current_input = layer(current_input, training=training)
            elif isinstance(layer, (Dense, Conv2D, BatchNorm, LayerNorm)):
                current_input = layer(current_input, training=training)
            else:
                # Other layers (Flatten, MaxPooling2D, etc.) just take X
                current_input = layer(current_input)

        return current_input

    def save(self, filename: str):
        """Save the model's parameters to a file."""
        import pickle

        params = []
        for layer in self.layers:
            if hasattr(layer, "params"):
                params.append(layer.params)
        with open(filename, "wb") as f:
            pickle.dump(params, f)

    def load(self, filename: str):
        """Load the model's parameters from a file."""
        import pickle

        with open(filename, "rb") as f:
            params = pickle.load(f)
        param_idx = 0
        for layer in self.layers:
            if hasattr(layer, "params"):
                layer.params = params[param_idx]
                param_idx += 1
