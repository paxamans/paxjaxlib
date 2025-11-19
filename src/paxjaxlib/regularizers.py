import jax.numpy as jnp


def l1(alpha=1.0):
    """L1 regularization."""

    def regularizer(x):
        return alpha * jnp.sum(jnp.abs(x))

    return regularizer


def l2(alpha=1.0):
    """L2 regularization."""

    def regularizer(x):
        return alpha * jnp.sum(jnp.square(x))

    return regularizer
