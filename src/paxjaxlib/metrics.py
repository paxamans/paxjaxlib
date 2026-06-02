"""
Metric functions for paxjaxlib.

Metrics are evaluated at the end of each training epoch by the
:class:`~paxjaxlib.training.Trainer`.  They follow the signature
``metric_fn(y_true, y_pred) → scalar``.
"""

import jax.numpy as jnp


def accuracy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """Classification accuracy.

    Handles both 1-D labels and one-hot encoded labels (uses ``argmax``).

    Args:
        y_true: Ground-truth labels.
        y_pred: Model predictions.
    """
    # For binary/multiclass: handle both 1D and 2D arrays
    if y_true.ndim == 1:
        return jnp.mean(y_pred == y_true)
    else:
        return jnp.mean(jnp.argmax(y_pred, axis=-1) == jnp.argmax(y_true, axis=-1))


def precision(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """Precision score for binary classification.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
    """
    true_positives = jnp.sum((y_pred == 1) & (y_true == 1))
    predicted_positives = jnp.sum(y_pred == 1)
    return true_positives / (predicted_positives + 1e-7)


def recall(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """Recall score for binary classification.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
    """
    true_positives = jnp.sum((y_pred == 1) & (y_true == 1))
    actual_positives = jnp.sum(y_true == 1)
    return true_positives / (actual_positives + 1e-7)


def f1_score(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """F1 score (harmonic mean of precision and recall).

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r + 1e-7)
