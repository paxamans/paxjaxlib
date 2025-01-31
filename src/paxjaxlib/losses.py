import jax.numpy as jnp

def mse_loss(y_pred, y_true, params=None, reg_lambda=0.01):
    loss = jnp.mean((y_pred - y_true) ** 2)
    if params is not None and reg_lambda > 0:
        l2_penalty = sum(jnp.sum(jnp.square(w)) for w, _ in params)
        loss += reg_lambda * l2_penalty
    return loss

def binary_crossentropy(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> float:
    epsilon = 1e-15
    y_pred = jnp.clip(y_pred, epsilon, 1 - epsilon)
    return -jnp.mean(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))

def categorical_crossentropy(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> float:
    epsilon = 1e-15
    y_pred = jnp.clip(y_pred, epsilon, 1 - epsilon)
    return -jnp.mean(jnp.sum(y_true * jnp.log(y_pred), axis=-1))
