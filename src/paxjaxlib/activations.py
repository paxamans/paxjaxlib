import jax.nn
import jax.numpy as jnp
import jax.scipy.special


def relu(x):
    return jnp.maximum(0, x)


def linear(x):
    return x


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def tanh(x):
    return jnp.tanh(x)


def softmax(x):
    x_max = jnp.max(x, axis=-1, keepdims=True)
    unnormalized = jnp.exp(x - x_max)
    return unnormalized / jnp.sum(unnormalized, axis=-1, keepdims=True)


def gelu(x):
    return x * 0.5 * (1.0 + jax.scipy.special.erf(x / jnp.sqrt(2.0)))


def silu(x):
    return x * sigmoid(x)


def mish(x):
    return jax.nn.mish(x)
