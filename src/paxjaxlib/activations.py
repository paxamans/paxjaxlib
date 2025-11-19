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
    return jnp.exp(x) / jnp.sum(jnp.exp(x), axis=-1, keepdims=True)


def gelu(x):
    return x * 0.5 * (1.0 + jax.scipy.special.erf(x / jnp.sqrt(2.0)))


def silu(x):
    return x * sigmoid(x)


def mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))
