"""
Regularisation functions for paxjaxlib.

Regularisers are factories that return a function
``regularizer(weights) → scalar_penalty``.  Pass them to layer
constructors::

    Dense(10, 64, key, kernel_regularizer=l2(0.01))
"""

import jax.numpy as jnp


def l1(alpha: float = 1.0):
    """L1 (Lasso) regularisation: ``alpha * sum(|w|)``.

    Args:
        alpha: Regularisation strength. Default ``1.0``.
    """

    def regularizer(x: jnp.ndarray) -> jnp.ndarray:
        return alpha * jnp.sum(jnp.abs(x))

    return regularizer


def l2(alpha: float = 1.0):
    """L2 (Ridge) regularisation: ``alpha * sum(w²)``.

    Args:
        alpha: Regularisation strength. Default ``1.0``.
    """

    def regularizer(x: jnp.ndarray) -> jnp.ndarray:
        return alpha * jnp.sum(jnp.square(x))

    return regularizer
