import jax.numpy as jnp
from jax import random

from paxjaxlib import initializers


def test_xavier_uniform():
    key = random.PRNGKey(0)
    shape = (100, 100)
    initializer = initializers.xavier_uniform()
    weights = initializer(key, shape)

    assert weights.shape == shape
    assert weights.min() >= -1.0
    assert weights.max() <= 1.0
    assert jnp.abs(weights.mean()) < 0.1


def test_he_normal():
    key = random.PRNGKey(0)
    shape = (100, 100)
    initializer = initializers.he_normal()
    weights = initializer(key, shape)

    assert weights.shape == shape
    assert jnp.abs(weights.mean()) < 0.1
    assert jnp.abs(weights.std() - jnp.sqrt(2 / 100)) < 0.1


def test_lecun_normal():
    key = random.PRNGKey(0)
    shape = (100, 100)
    initializer = initializers.lecun_normal()
    weights = initializer(key, shape)

    assert weights.shape == shape
    assert jnp.abs(weights.mean()) < 0.1
