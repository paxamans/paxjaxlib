import jax.numpy as jnp
import pytest
from jax import random

from paxjaxlib.layers import Conv2D, Dense, Dropout, Flatten


def test_dense_layer():
    key = random.PRNGKey(0)
    input_dim = 10
    output_dim = 5
    batch_size = 32
    
    layer = Dense(input_dim, output_dim, key)
    
    X = random.normal(key, (batch_size, input_dim))
    output = layer(X)
    
    assert output.shape == (batch_size, output_dim)
    assert layer.W.shape == (input_dim, output_dim)
    assert layer.b.shape == (output_dim,)

def test_conv2d_layer():
    key = random.PRNGKey(0)
    input_channels = 3
    output_channels = 8
    kernel_size = (3, 3)
    batch_size = 4
    height = 28
    width = 28
    
    layer = Conv2D(input_channels, output_channels, kernel_size, key)
    
    X = random.normal(key, (batch_size, height, width, input_channels))
    output = layer(X)
    
    assert output.shape == (batch_size, height, width, output_channels)
    assert layer.W.shape == (kernel_size[0], kernel_size[1], input_channels, output_channels)

def test_dropout_layer():
    key = random.PRNGKey(0)
    rate = 0.5
    layer = Dropout(rate)
    
    X = jnp.ones((10, 10))
    
    # Test training mode (should drop some values)
    output_train = layer(X, key=key, training=True)
    assert not jnp.allclose(output_train, X)
    
    # Test eval mode (should be identity)
    output_eval = layer(X, training=False)
    assert jnp.allclose(output_eval, X)

def test_flatten_layer():
    layer = Flatten()
    X = jnp.ones((10, 28, 28, 1))
    output = layer(X)
    assert output.shape == (10, 28 * 28 * 1)
