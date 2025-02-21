import jax.numpy as jnp
from jax import random
from typing import Callable, Tuple
import jax.lax as lax

class Conv2D:
    """2D Convolutional Layer"""
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: Tuple[int, int],
        activation: Callable,
        key: random.PRNGKey,
        stride: Tuple[int, int] = (1, 1),
        padding: str = "SAME",
    ):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.stride = stride
        self.padding = padding

        # Initialize parameters
        key_W, key_b = random.split(key)
        self.W = random.normal(
            key_W, 
            (kernel_size[0], kernel_size[1], input_channels, output_channels)
        ) * 0.01
        self.b = jnp.zeros((output_channels,))

    def forward(self, X: jnp.ndarray, W=None, b=None) -> jnp.ndarray:
        W = self.W if W is None else W
        b = self.b if b is None else b
        conv_output = lax.conv_general_dilated(
            X,
            W,
            window_strides=self.stride,
            padding=self.padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=1,
        )
        return self.activation(conv_output + b[None, None, None, :])

    @property
    def parameters(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return (self.W, self.b)

    @parameters.setter
    def parameters(self, params: Tuple[jnp.ndarray, jnp.ndarray]):
        self.W, self.b = params


class Flatten:
    """Flatten Layer for shape transition between Conv2D and Dense layers"""
    def __init__(self):
        pass

    def forward(self, X: jnp.ndarray) -> jnp.ndarray:
        return X.reshape(X.shape[0], -1)

    @property
    def parameters(self) -> Tuple:
        return ()

    @parameters.setter
    def parameters(self, params: Tuple):
        pass

class Dense:
    """Dense layer"""
    def __init__(self, input_dim: int, output_dim: int, activation: Callable, key: random.PRNGKey):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        
        # Initialize parameters
        key_W, key_b = random.split(key)
        self.W = random.normal(key_W, (input_dim, output_dim)) * 0.01
        self.b = jnp.zeros((output_dim,))
        
    def forward(self, X: jnp.ndarray, W=None, b=None) -> jnp.ndarray:
        W = self.W if W is None else W
        b = self.b if b is None else b
        return self.activation(jnp.dot(X, W)) + b

    @property
    def parameters(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return (self.W, self.b)
    
    @parameters.setter
    def parameters(self, params: Tuple[jnp.ndarray, jnp.ndarray]):
        self.W, self.b = params
