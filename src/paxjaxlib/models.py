from typing import Sequence, Callable, List
import jax.numpy as jnp
from .layers import Dense, Conv2D, Flatten  
from jax import tree_util, random
import pickle

class NeuralNetwork:
    def __init__(self, layers: List):
        """
        Initialize the neural network with a list of layers.

        Args:
            layers (List): A list of layer instances (e.g., Dense, Conv2D, Flatten).
        """
        self.layers = layers
        self.activations = []
        for layer in self.layers:
            if hasattr(layer, 'activation'):
                self.activations.append(layer.activation)
            else:
                # For layers without activation (e.g., Flatten), use identity
                self.activations.append(lambda x: x)

    def forward(self, X):
        """
        Forward pass through the network.

        Args:
            X (jnp.ndarray): Input data.

        Returns:
            jnp.ndarray: Output of the network.
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def save(self, filename: str):
        """
        Save the model parameters to a file.

        Args:
            filename (str): Path to the file where parameters will be saved.
        """
        params = self.parameters
        with open(filename, 'wb') as f:
            pickle.dump(params, f)

    @classmethod
    def load(cls, layers: List, filename: str, key: random.PRNGKey = None):
        """
        Load the model parameters from a file.

        Args:
            layers (List): A list of layer instances (e.g., Dense, Conv2D, Flatten).
            filename (str): Path to the file from which parameters will be loaded.
            key (random.PRNGKey): Random key for parameter initialization (optional).

        Returns:
            NeuralNetwork: A model instance with loaded parameters.
        """
        with open(filename, 'rb') as f:
            params = pickle.load(f)
        model = cls(layers)
        model.parameters = params
        return model

    @property
    def parameters(self):
        """
        Get the parameters of all layers that have parameters.
        """
        return [layer.parameters for layer in self.layers if hasattr(layer, 'parameters') and layer.parameters]

    @parameters.setter
    def parameters(self, params):
        """
        Set the parameters of all layers that have parameters.
        """
        param_idx = 0
        for layer in self.layers:
            if hasattr(layer, 'parameters') and layer.parameters:
                layer.parameters = params[param_idx]
                param_idx += 1
