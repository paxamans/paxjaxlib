from typing import Callable, Optional, Tuple, Union

import jax.lax as lax
import jax.numpy as jnp
from jax import random

from .core import Module


class Dropout(Module):
    """Dropout Layer"""
    def __init__(self, rate: float):
        """
        Initialize the Dropout layer.

        Args:
            rate (float): Fraction of the input units to drop.
        """
        if not (0.0 <= rate < 1.0):
            raise ValueError("Dropout rate must be in the interval [0, 1).")
        self.rate = rate

    def __call__(self, X: jnp.ndarray, key: Optional[random.PRNGKey] = None, training: bool = False) -> jnp.ndarray:
        """
        Apply dropout.

        Args:
            X (jnp.ndarray): Input data.
            key (Optional[random.PRNGKey]): JAX PRNGKey for dropout. Required if training is True.
            training (bool): If True, applies dropout.
        """
        if not training or self.rate == 0.0:
            return X

        if key is None:
            raise ValueError("Dropout layer requires a PRNGKey during training.")

        keep_prob = 1.0 - self.rate
        mask = random.bernoulli(key, p=keep_prob, shape=X.shape)
        return (X * mask) / keep_prob


class Conv2D(Module):
    """2D Convolutional Layer"""
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: Tuple[int, int],
        key: random.PRNGKey,
        activation: Callable = lambda x: x,
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

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        conv_output = lax.conv_general_dilated(
            X,
            self.W,
            window_strides=self.stride,
            padding=self.padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=1,
        )
        return self.activation(conv_output + self.b[None, None, None, :])


class Flatten(Module):
    """Flatten Layer"""
    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return X.reshape(X.shape[0], -1)


class Dense(Module):
    """Dense layer"""
    def __init__(self, input_dim: int, output_dim: int, key: random.PRNGKey, activation: Callable = lambda x: x):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        
        # Initialize parameters
        key_W, key_b = random.split(key)
        self.W = random.normal(key_W, (input_dim, output_dim)) * 0.01
        self.b = jnp.zeros((output_dim,))
        
    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return self.activation(jnp.dot(X, self.W) + self.b)
