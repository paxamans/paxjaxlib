from typing import Sequence, Callable
import jax.numpy as jnp
from .layers import Layer

class NeuralNetwork:
    def __init__(self, layer_sizes: Sequence[int], activations: Sequence[Callable]):
        assert len(layer_sizes) - 1 == len(activations), "Invalid layer configuration"
        
        self.layers = []
        self._activations = activations  # Store activations
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i + 1], activations[i])
            self.layers.append(layer)
    
    def forward(self, X: jnp.ndarray) -> jnp.ndarray:
        current_input = X
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input
    
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
