import jax.numpy as jnp


def mse_loss(y_pred, y_true):
    return jnp.mean((y_pred - y_true) ** 2)


def mean_squared_error(y_pred, y_true):
    return jnp.mean((y_pred - y_true) ** 2)


def binary_crossentropy(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    epsilon = 1e-15
    y_pred = jnp.clip(y_pred, epsilon, 1 - epsilon)
    return -jnp.mean(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))


def categorical_crossentropy(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    epsilon = 1e-15
    y_pred = jnp.clip(y_pred, epsilon, 1 - epsilon)
    return -jnp.mean(jnp.sum(y_true * jnp.log(y_pred), axis=-1))


def huber_loss(y_pred, y_true, delta=1.0):
    """Huber loss function."""
    error = y_pred - y_true
    is_small_error = jnp.abs(error) <= delta
    squared_loss = jnp.square(error) / 2
    linear_loss = delta * (jnp.abs(error) - delta / 2)
    return jnp.where(is_small_error, squared_loss, linear_loss).mean()


def hinge_loss(y_pred, y_true):
    """Hinge loss function."""
    return jnp.mean(jnp.maximum(0, 1 - y_pred * y_true))
