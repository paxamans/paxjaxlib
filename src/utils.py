import jax.numpy as jnp

def relu(x):
    return jnp.maximum(0, x)

def linear(x):
    return x
