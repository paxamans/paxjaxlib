from typing import Sequence, Callable
import jax.numpy as jnp
from .layers import Layer
from jax import tree_util, random
import pickle

class NeuralNetwork:
    def __init__(self, layer_sizes: Sequence[int], activations: Sequence[Callable], key: random.PRNGKey):
        assert len(layer_sizes) - 1 == len(activations), "Invalid layer configuration"
        
        self.layers = []
        self._activations = activations
        
        # Generate keys for each layer
        keys = random.split(key, len(activations))
        
        for i in range(len(layer_sizes) - 1):
            layer = Layer(
                layer_sizes[i], 
                layer_sizes[i + 1], 
                activations[i], 
                key=keys[i]
            )
            self.layers.append(layer)

    def forward(self, X: jnp.ndarray) -> jnp.ndarray:
        current_input = X
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input

    def save(self, filename: str):
        params = self.parameters
        with open(filename, 'wb') as f:
            pickle.dump(params, f)

    @classmethod
    def load(cls, layer_sizes, activations, filename: str, key):
        with open(filename, 'rb') as f:
            params = pickle.load(f)
        model = cls(layer_sizes, activations, key)
        model.parameters = params
        return model

    
    @property
    def parameters(self):
        return [layer.parameters for layer in self.layers]
    
    @parameters.setter
    def parameters(self, params):
        for layer, param in zip(self.layers, params):
            layer.parameters = param
            
    @property
    def activations(self):
        return self._activations
