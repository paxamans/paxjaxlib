import jax.numpy as jnp
from jax import random


def _compute_fans(shape):
    """Computes the number of input and output units for a weight shape."""
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


def xavier_uniform(gain=1.0, dtype=jnp.float32):
    """Xavier uniform initializer."""

    def initializer(key, shape):
        fan_in, fan_out = _compute_fans(shape)
        bound = gain * jnp.sqrt(6.0 / (fan_in + fan_out))
        return random.uniform(key, shape, dtype, -bound, bound)

    return initializer


def he_normal(gain=1.0, dtype=jnp.float32):
    """He normal initializer."""

    def initializer(key, shape):
        fan_in, _ = _compute_fans(shape)
        std = gain / jnp.sqrt(fan_in)
        return std * random.normal(key, shape, dtype)

    return initializer


def lecun_normal(gain=1.0, dtype=jnp.float32):
    """LeCun normal initializer."""

    def initializer(key, shape):
        fan_in, _ = _compute_fans(shape)
        std = gain / jnp.sqrt(fan_in)
        return std * random.normal(key, shape, dtype)

    return initializer
