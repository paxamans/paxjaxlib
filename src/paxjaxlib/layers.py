import jax.numpy as jnp
from jax import random
from typing import Callable, Tuple

class Layer:
    def __init__(self, input_dim: int, output_dim: int, activation: Callable, key: random.PRNGKey):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        
        # Split the key for W and b initialization
        key_W, key_b = random.split(key)
        self.W = random.normal(key_W, (input_dim, output_dim)) * 0.01
        self.b = jnp.zeros((output_dim,))
        
    def forward(self, X: jnp.ndarray) -> jnp.ndarray:
        return self.activation(jnp.dot(X, self.W) + self.b)
    
    @property
    def parameters(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return (self.W, self.b)
    
    @parameters.setter
    def parameters(self, params: Tuple[jnp.ndarray, jnp.ndarray]):
        self.W, self.b = params
