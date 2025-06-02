import jax.numpy as jnp

def mse_loss(y_pred, y_true, params=None, reg_lambda=0.01):
    loss = jnp.mean((y_pred - y_true) ** 2)
    if params is not None and reg_lambda > 0:
        # This L2 penalty assumes params is a list of tuples (W, b) or just (W,)
        # It also tries to handle cases where a layer might not have W or b (e.g. () for Dropout)
        l2_penalty = 0.0
        for p_tuple in params:
            if p_tuple: # If the parameter tuple is not empty
                for p_item in p_tuple: # Iterate over W, b etc. in the tuple
                    if p_item is not None:
                        l2_penalty += jnp.sum(jnp.square(p_item))
        loss += reg_lambda * l2_penalty
    return loss

def binary_crossentropy(y_pred: jnp.ndarray, y_true: jnp.ndarray, params=None, reg_lambda=0.0) -> float:
    epsilon = 1e-15
    y_pred = jnp.clip(y_pred, epsilon, 1 - epsilon)
    loss = -jnp.mean(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))
    if params is not None and reg_lambda > 0:
        l2_penalty = 0.0
        for p_tuple in params:
            if p_tuple:
                for p_item in p_tuple:
                    if p_item is not None:
                        l2_penalty += jnp.sum(jnp.square(p_item))
        loss += reg_lambda * l2_penalty
    return loss

def categorical_crossentropy(y_pred: jnp.ndarray, y_true: jnp.ndarray, params=None, reg_lambda=0.0) -> float:
    epsilon = 1e-15
    y_pred = jnp.clip(y_pred, epsilon, 1 - epsilon)
    loss = -jnp.mean(jnp.sum(y_true * jnp.log(y_pred), axis=-1)) # Main loss calculation
    if params is not None and reg_lambda > 0:
        l2_penalty = 0.0
        for p_tuple in params: # params is model.parameters, a list of tuples
            if p_tuple: # Check if the layer's parameter tuple is not empty (e.g. () for Dropout)
                for p_item in p_tuple: # Iterate over W, b etc. in the tuple
                    if p_item is not None: # Check if the specific parameter (W or b) exists
                        l2_penalty += jnp.sum(jnp.square(p_item))
        loss += reg_lambda * l2_penalty
    return loss
