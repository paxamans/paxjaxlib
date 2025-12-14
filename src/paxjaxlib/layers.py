from typing import Any, Callable, Optional, Tuple, Union, cast

import jax.lax as lax
import jax.numpy as jnp
from jax import random

from . import activations
from .core import Module
from .initializers import he_normal


class Dropout(Module):
    """Dropout Layer"""

    def __init__(self, rate: float):
        """
        Initialize the Dropout layer.

        Args:
            rate (float): Fraction of the input units to drop.
        """
        super().__init__()
        if not (0.0 <= rate < 1.0):
            raise ValueError("Dropout rate must be in the interval [0, 1).")
        self.rate = rate

    def __call__(
        self,
        X: jnp.ndarray,
        key: Optional[Any] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        """
        Apply dropout.

        Args:
            X (jnp.ndarray): Input data.
            key (Optional[Any]): JAX PRNGKey for dropout. Required if training is True.
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
        key: Any,
        activation: Callable = lambda x: x,
        stride: Tuple[int, int] = (1, 1),
        padding: str = "SAME",
        kernel_initializer: Optional[Callable] = None,
        bias_initializer: Optional[Callable] = None,
        kernel_regularizer: Optional[Callable] = None,
        bias_regularizer: Optional[Callable] = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.stride = stride
        self.padding = padding
        self.kernel_initializer = kernel_initializer or he_normal()
        self.bias_initializer = bias_initializer or (
            lambda key, shape: jnp.zeros(shape)
        )
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        # Initialize weights and biases
        self.W = self.kernel_initializer(
            key,
            (
                self.kernel_size[0],
                self.kernel_size[1],
                self.input_channels,
                self.output_channels,
            ),
        )
        self.b = self.bias_initializer(key, (self.output_channels,))

    @property
    def params(self):
        return {"W": self.W, "b": self.b}

    @params.setter
    def params(self, value):
        if isinstance(value, dict):
            self.W = value["W"]
            self.b = value["b"]

    def __call__(self, X: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if training:
            if self.kernel_regularizer:
                self.add_loss(self.kernel_regularizer(self.W))
            if self.bias_regularizer:
                self.add_loss(self.bias_regularizer(self.b))
        conv_output = lax.conv_general_dilated(
            X,
            self.W,
            window_strides=self.stride,
            padding=self.padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=1,
        )
        return cast(
            jnp.ndarray, self.activation(conv_output + self.b[None, None, None, :])
        )


class Flatten(Module):
    """Flatten Layer"""

    def __init__(self):
        super().__init__()

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return X.reshape(X.shape[0], -1)


class MaxPooling2D(Module):
    """Max Pooling 2D Layer"""

    def __init__(
        self,
        pool_size: Tuple[int, int],
        strides: Optional[Tuple[int, int]] = None,
        padding: str = "VALID",
    ):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides or pool_size
        self.padding = padding

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return cast(
            jnp.ndarray,
            lax.reduce_window(
                X,
                -jnp.inf,
                lax.max,
                (1, *self.pool_size, 1),
                (1, *self.strides, 1),
                self.padding,
            ),
        )


class AvgPooling2D(Module):
    """Average Pooling 2D Layer"""

    def __init__(
        self,
        pool_size: Tuple[int, int],
        strides: Optional[Tuple[int, int]] = None,
        padding: str = "VALID",
    ):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides or pool_size
        self.padding = padding

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        counts = lax.reduce_window(
            jnp.ones_like(X),
            0.0,
            lax.add,
            (1, *self.pool_size, 1),
            (1, *self.strides, 1),
            self.padding,
        )
        sums = lax.reduce_window(
            X,
            0.0,
            lax.add,
            (1, *self.pool_size, 1),
            (1, *self.strides, 1),
            self.padding,
        )
        return cast(jnp.ndarray, sums / counts)


class BatchNorm(Module):
    """Batch Normalization Layer"""

    def __init__(self, input_dim: int, key: Any, momentum=0.99, epsilon=1e-5):
        super().__init__()
        self.input_dim = input_dim
        self.momentum = momentum
        self.epsilon = epsilon
        self.gamma = jnp.ones(input_dim)
        self.beta = jnp.zeros(input_dim)
        self.running_mean = jnp.zeros(input_dim)
        self.running_var = jnp.ones(input_dim)

    @property
    def params(self):
        return {
            "gamma": self.gamma,
            "beta": self.beta,
            "running_mean": self.running_mean,
            "running_var": self.running_var,
        }

    @params.setter
    def params(self, value):
        if isinstance(value, dict):
            self.gamma = value["gamma"]
            self.beta = value["beta"]
            self.running_mean = value["running_mean"]
            self.running_var = value["running_var"]

    def __call__(self, X: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if training:
            mean = jnp.mean(X, axis=0)
            var = jnp.var(X, axis=0)
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )
        else:
            mean = self.running_mean
            var = self.running_var
        return self.gamma * (X - mean) / jnp.sqrt(var + self.epsilon) + self.beta


class LayerNorm(Module):
    """Layer Normalization Layer"""

    def __init__(self, shape: Optional[Tuple] = None, epsilon=1e-5):
        super().__init__()
        self.shape = shape
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None
        if shape is not None:
            if isinstance(shape, tuple):
                self.gamma = jnp.ones(shape)
                self.beta = jnp.zeros(shape)
            else:
                self.gamma = jnp.ones(shape)
                self.beta = jnp.zeros(shape)

    @property
    def params(self):
        return {"gamma": self.gamma, "beta": self.beta}

    @params.setter
    def params(self, value):
        if isinstance(value, dict):
            self.gamma = value["gamma"]
            self.beta = value["beta"]

    def build(self, input_shape, key):
        if self.gamma is None:
            self.gamma = jnp.ones(input_shape[-1])
            self.beta = jnp.zeros(input_shape[-1])

    def __call__(self, X: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if self.gamma is None:
            # Initialize on first call if not already initialized
            if self.shape is None:
                self.gamma = jnp.ones(X.shape[-1:])
                self.beta = jnp.zeros(X.shape[-1:])
        mean = jnp.mean(X, axis=-1, keepdims=True)
        var = jnp.var(X, axis=-1, keepdims=True)
        if self.gamma is None or self.beta is None:
            raise ValueError("LayerNorm not initialized")
        return self.gamma * (X - mean) / jnp.sqrt(var + self.epsilon) + self.beta


class Dense(Module):
    """Dense layer"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        key: Any,
        activation: Union[Callable, str, None] = None,
        kernel_initializer: Optional[Callable] = None,
        bias_initializer: Optional[Callable] = None,
        kernel_regularizer: Optional[Callable] = None,
        bias_regularizer: Optional[Callable] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Handle string activations
        if isinstance(activation, str):
            self.activation = getattr(activations, activation.lower())
        else:
            self.activation = activation if activation is not None else lambda x: x
        self.kernel_initializer = kernel_initializer or he_normal()
        self.bias_initializer = bias_initializer or (
            lambda key, shape: jnp.zeros(shape)
        )
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        # Initialize weights and biases immediately
        self.W = self.kernel_initializer(key, (self.input_dim, self.output_dim))
        self.b = self.bias_initializer(key, (self.output_dim,))
        self.built = True

    @property
    def params(self):
        """Return parameters as a dictionary."""
        return {"W": self.W, "b": self.b}

    @params.setter
    def params(self, value):
        """Set parameters from a dictionary."""
        if isinstance(value, dict):
            self.W = value["W"]
            self.b = value["b"]

    def __call__(self, X: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if training:
            if self.kernel_regularizer:
                self.add_loss(self.kernel_regularizer(self.W))
            if self.bias_regularizer:
                self.add_loss(self.bias_regularizer(self.b))
        Z = jnp.dot(X, self.W) + self.b
        if self.activation is not None:
            return cast(jnp.ndarray, self.activation(Z))
        return Z
