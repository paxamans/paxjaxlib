import pickle
from typing import List, Optional

import jax.numpy as jnp
from jax import random

from .core import Module
from .layers import Dropout


class NeuralNetwork(Module):
    def __init__(self, layers: List[Module]):
        self.layers = layers

    def __call__(self, X: jnp.ndarray, key: Optional[random.PRNGKey] = None, training: bool = False) -> jnp.ndarray:
        """
        Forward pass through the network.
        """
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
            else:
                # Other layers (Conv2D, Dense, Flatten) just take X
                # If we add more layers with state/randomness, we'd handle them here
                current_input = layer(current_input)
                
        return current_input

    def save(self, filename: str):
        """Save the entire model (as a Pytree) using pickle."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str):
        """Load the entire model."""
        with open(filename, 'rb') as f:
            return pickle.load(f)