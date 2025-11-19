import jax.numpy as jnp
from jax import random


def xavier_uniform(gain=1.0, dtype=jnp.float32):
    """Xavier uniform initializer."""

    def initializer(key, shape):
        if len(shape) >= 2:
            fan_in, fan_out = shape[0], shape[-1]
        else:
            fan_in = fan_out = shape[0] if len(shape) == 1 else 1
        bound = gain * jnp.sqrt(6.0 / (fan_in + fan_out))
        return random.uniform(key, shape, dtype, -bound, bound)

    return initializer


def he_normal(gain=1.0, dtype=jnp.float32):
    """He normal initializer."""

    def initializer(key, shape):
        fan_in = shape[0] if len(shape) >= 1 else 1
        std = gain / jnp.sqrt(fan_in)
        return std * random.normal(key, shape, dtype)

    return initializer


def lecun_normal(gain=1.0, dtype=jnp.float32):
    """LeCun normal initializer."""

    def initializer(key, shape):
        fan_in = shape[0] if len(shape) >= 1 else 1
        std = gain / jnp.sqrt(fan_in)
        return std * random.normal(key, shape, dtype)

    return initializer
