"""
Loss functions for paxjaxlib.

All losses follow the signature ``loss_fn(y_pred, y_true) -> scalar``
so they can be passed directly to :class:`paxjaxlib.training.Trainer`.
"""

import jax.numpy as jnp


def mse_loss(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """Mean Squared Error loss.

    Args:
        y_pred: Predicted values.
        y_true: Ground-truth values.
    """
    return jnp.mean((y_pred - y_true) ** 2)


def binary_crossentropy(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """Binary cross-entropy loss.

    Clips predictions to ``[epsilon, 1 - epsilon]`` for numerical stability.

    Args:
        y_pred: Predicted probabilities in ``[0, 1]``.
        y_true: Binary ground-truth labels.
    """
    epsilon = 1e-15
    y_pred = jnp.clip(y_pred, epsilon, 1 - epsilon)
    return -jnp.mean(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))


def categorical_crossentropy(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """Categorical cross-entropy loss for one-hot encoded labels.

    Args:
        y_pred: Predicted probabilities (e.g. after softmax).
        y_true: One-hot encoded ground-truth labels.
    """
    epsilon = 1e-15
    y_pred = jnp.clip(y_pred, epsilon, 1 - epsilon)
    return -jnp.mean(jnp.sum(y_true * jnp.log(y_pred), axis=-1))


def huber_loss(
    y_pred: jnp.ndarray, y_true: jnp.ndarray, delta: float = 1.0
) -> jnp.ndarray:
    """Huber loss — quadratic for small errors, linear for large ones.

    Args:
        y_pred: Predicted values.
        y_true: Ground-truth values.
        delta: Threshold at which to switch from quadratic to linear. Default ``1.0``.
    """
    error = y_pred - y_true
    is_small_error = jnp.abs(error) <= delta
    squared_loss = jnp.square(error) / 2
    linear_loss = delta * (jnp.abs(error) - delta / 2)
    return jnp.where(is_small_error, squared_loss, linear_loss).mean()


def hinge_loss(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """Hinge loss for binary classification with labels in ``{-1, +1}``.

    Args:
        y_pred: Raw model outputs (not probabilities).
        y_true: Ground-truth labels in ``{-1, +1}``.
    """
    return jnp.mean(jnp.maximum(0, 1 - y_pred * y_true))


def cosine_similarity_loss(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """Cosine similarity loss: ``1 - cosine_similarity``.

    Returns ``0`` when vectors are identical, ``2`` when opposite.

    Args:
        y_pred: Predicted embedding vectors.
        y_true: Target embedding vectors.
    """
    y_pred_norm = y_pred / (jnp.linalg.norm(y_pred, axis=-1, keepdims=True) + 1e-8)
    y_true_norm = y_true / (jnp.linalg.norm(y_true, axis=-1, keepdims=True) + 1e-8)
    return jnp.mean(1.0 - jnp.sum(y_pred_norm * y_true_norm, axis=-1))


def kl_divergence(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """Kullback-Leibler divergence: ``KL(y_true || y_pred)``.

    Both inputs should be valid probability distributions (summing to 1
    along the last axis).

    Args:
        y_pred: Predicted probability distribution.
        y_true: Target probability distribution.
    """
    epsilon = 1e-15
    y_pred = jnp.clip(y_pred, epsilon, 1.0)
    y_true = jnp.clip(y_true, epsilon, 1.0)
    return jnp.mean(jnp.sum(y_true * jnp.log(y_true / y_pred), axis=-1))
