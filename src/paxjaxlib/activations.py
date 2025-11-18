import jax.numpy as jnp


def relu(x):
    return jnp.maximum(0, x)

def linear(x):
    return x

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def tanh(x):
    return jnp.tanh(x)

def softmax(x):
    return jnp.exp(x) / jnp.sum(jnp.exp(x), axis=-1, keepdims=True)
