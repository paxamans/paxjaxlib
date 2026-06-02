"""
Weight initialisation functions for paxjaxlib.

Each initialiser is a factory that returns a function with signature
``(key, shape) → jnp.ndarray``, so they can be passed to layer
constructors::

    Dense(10, 64, key, kernel_initializer=he_normal())
"""

import jax.numpy as jnp
from jax import random


def _compute_fans(shape: tuple) -> tuple[int, int]:
    """Compute the number of input and output units for a weight shape.

    For 2-D shapes ``(fan_in, fan_out)``.  For convolution kernels
    in HWIO format, the receptive field is folded into the fan counts.
    """
    if len(shape) < 1:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolution kernels (HWIO).
        # shape = (H, W, In, Out)
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out


def xavier_uniform(gain: float = 1.0, dtype=jnp.float32):
    """Xavier / Glorot uniform initialiser.

    Draws weights from ``U[-bound, bound]`` where
    ``bound = gain * sqrt(6 / (fan_in + fan_out))``.

    Args:
        gain: Multiplicative scaling factor. Default ``1.0``.
        dtype: Data type of the resulting array.
    """

    def initializer(key, shape):
        fan_in, fan_out = _compute_fans(shape)
        bound = gain * jnp.sqrt(6.0 / (fan_in + fan_out))
        return random.uniform(key, shape, dtype, -bound, bound)

    return initializer


def he_normal(gain: float = 1.0, dtype=jnp.float32):
    """He (Kaiming) normal initialiser.

    Draws weights from ``N(0, gain / sqrt(fan_in))``.  Best used with
    ReLU activations.

    Args:
        gain: Multiplicative scaling factor. Default ``1.0``.
        dtype: Data type of the resulting array.
    """

    def initializer(key, shape):
        fan_in, _ = _compute_fans(shape)
        std = gain / jnp.sqrt(fan_in)
        return std * random.normal(key, shape, dtype)

    return initializer


def lecun_normal(gain: float = 1.0, dtype=jnp.float32):
    """LeCun normal initialiser.

    Draws weights from ``N(0, gain / sqrt(fan_in))``.  Recommended for
    SELU activations.

    Args:
        gain: Multiplicative scaling factor. Default ``1.0``.
        dtype: Data type of the resulting array.
    """

    def initializer(key, shape):
        fan_in, _ = _compute_fans(shape)
        std = gain / jnp.sqrt(fan_in)
        return std * random.normal(key, shape, dtype)

    return initializer
