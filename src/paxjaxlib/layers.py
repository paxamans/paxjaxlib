import jax.numpy as jnp
from jax import random
from typing import Callable, Tuple, Optional
import jax.lax as lax

class Dropout:
    """Dropout Layer"""
    def __init__(self, rate: float):
        """
        Initialize the Dropout layer.

        Args:
            rate (float): Fraction of the input units to drop (e.g., 0.5 means 50% drop).
                          Must be in the interval [0, 1).
        """
        if not (0.0 <= rate < 1.0):
            raise ValueError("Dropout rate must be in the interval [0, 1). rate=1.0 would zero everything.")
        self.rate = rate

    def forward(self, X: jnp.ndarray, training: bool = False, key: Optional[random.PRNGKey] = None) -> jnp.ndarray:
        """
        Apply dropout. Uses inverted dropout.

        Args:
            X (jnp.ndarray): Input data.
            training (bool): If True, applies dropout. Otherwise, returns the input as is.
            key (Optional[random.PRNGKey]): JAX PRNGKey for dropout.
                                             Required if training is True and rate > 0.

        Returns:
            jnp.ndarray: Output after applying dropout (if training) or the original input.
        """
        if not training or self.rate == 0.0:
            return X

        if key is None: # Key is required only if training and rate > 0
            raise ValueError("Dropout layer requires a PRNGKey for the forward pass during training when rate > 0.")

        keep_prob = 1.0 - self.rate
        # Create a mask by drawing from a Bernoulli distribution
        mask = random.bernoulli(key, p=keep_prob, shape=X.shape)
        # Apply mask and scale up during training (inverted dropout)
        return (X * mask) / keep_prob

    @property
    def parameters(self) -> Tuple: # Dropout has no learnable parameters
        return ()

    @parameters.setter
    def parameters(self, params: Tuple): # No parameters to set
        pass

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
        return self.activation(jnp.dot(X, W) + b)

    @property
    def parameters(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return (self.W, self.b)
    
    @parameters.setter
    def parameters(self, params: Tuple[jnp.ndarray, jnp.ndarray]):
        self.W, self.b = params
