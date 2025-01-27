import jax.numpy as jnp
from jax import random
from typing import Callable, Tuple

class Layer:
    def __init__(self, input_dim: int, output_dim: int, activation: Callable = lambda x: x):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        
        # Initialize parameters
        self.key = random.PRNGKey(0)
        self.W = random.normal(self.key, (input_dim, output_dim)) * 0.01 # weight
        self.b = jnp.zeros(output_dim) # bias
        
    def forward(self, X: jnp.ndarray) -> jnp.ndarray:
        return self.activation(jnp.dot(X, self.W) + self.b)
    
    @property
    def parameters(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return (self.W, self.b)
    
    @parameters.setter
    def parameters(self, params: Tuple[jnp.ndarray, jnp.ndarray]):
        self.W, self.b = params
