import jax.numpy as jnp


def accuracy(y_true, y_pred):
    """Classification accuracy."""
    # For binary/multiclass: handle both 1D and 2D arrays
    if y_true.ndim == 1:
        return jnp.mean(y_pred == y_true)
    else:
        return jnp.mean(jnp.argmax(y_pred, axis=-1) == jnp.argmax(y_true, axis=-1))


def precision(y_true, y_pred):
    """Precision score."""
    true_positives = jnp.sum((y_pred == 1) & (y_true == 1))
    predicted_positives = jnp.sum(y_pred == 1)
    return true_positives / (predicted_positives + 1e-7)


def recall(y_true, y_pred):
    """Recall score."""
    true_positives = jnp.sum((y_pred == 1) & (y_true == 1))
    actual_positives = jnp.sum(y_true == 1)
    return true_positives / (actual_positives + 1e-7)


def f1_score(y_true, y_pred):
    """F1 score."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r + 1e-7)
