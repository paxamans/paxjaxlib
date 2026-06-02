"""
Activation functions for paxjaxlib.

All activations are pure functions that operate element-wise on JAX arrays.
They can be passed directly to layer constructors (e.g. ``Dense(..., activation=relu)``)
or used as standalone layers in a ``NeuralNetwork`` layer list.
"""

import jax.nn
import jax.numpy as jnp
import jax.scipy.special


def relu(x: jnp.ndarray) -> jnp.ndarray:
    """Rectified Linear Unit: ``max(0, x)``."""
    return jnp.maximum(0, x)


def leaky_relu(x: jnp.ndarray, negative_slope: float = 0.01) -> jnp.ndarray:
    """Leaky ReLU: ``x`` if ``x > 0``, else ``negative_slope * x``.

    Args:
        x: Input array.
        negative_slope: Slope for negative values. Default ``0.01``.
    """
    return jnp.where(x > 0, x, negative_slope * x)


def elu(x: jnp.ndarray, alpha: float = 1.0) -> jnp.ndarray:
    """Exponential Linear Unit: ``x`` if ``x > 0``, else ``alpha * (exp(x) - 1)``.

    Args:
        x: Input array.
        alpha: Scale for the negative part. Default ``1.0``.
    """
    return jnp.where(x > 0, x, alpha * (jnp.exp(x) - 1))


def linear(x: jnp.ndarray) -> jnp.ndarray:
    """Identity activation (pass-through)."""
    return x


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """Sigmoid: ``1 / (1 + exp(-x))``."""
    return 1 / (1 + jnp.exp(-x))


def tanh(x: jnp.ndarray) -> jnp.ndarray:
    """Hyperbolic tangent activation."""
    return jnp.tanh(x)


def softmax(x: jnp.ndarray) -> jnp.ndarray:
    """Numerically-stable softmax over the last axis.

    Args:
        x: Input array of logits.
    """
    x_max = jnp.max(x, axis=-1, keepdims=True)
    unnormalized = jnp.exp(x - x_max)
    return unnormalized / jnp.sum(unnormalized, axis=-1, keepdims=True)


def gelu(x: jnp.ndarray) -> jnp.ndarray:
    """Gaussian Error Linear Unit (exact form)."""
    return x * 0.5 * (1.0 + jax.scipy.special.erf(x / jnp.sqrt(2.0)))


def silu(x: jnp.ndarray) -> jnp.ndarray:
    """SiLU / Swish activation: ``x * sigmoid(x)``."""
    return x * sigmoid(x)


def mish(x: jnp.ndarray) -> jnp.ndarray:
    """Mish activation: ``x * tanh(softplus(x))``."""
    return jnp.asarray(jax.nn.mish(x))
