import jax.numpy as jnp
from jax import random

from paxjaxlib import initializers, regularizers
from paxjaxlib.layers import (
    AvgPooling2D,
    BatchNorm,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    LayerNorm,
    MaxPooling2D,
)


def test_dense_layer_initializers_regularizers():
    key = random.PRNGKey(0)
    input_dim = 10
    output_dim = 5
    batch_size = 32

    layer = Dense(
        input_dim,
        output_dim,
        key,
        kernel_initializer=initializers.he_normal(),
        bias_initializer=initializers.xavier_uniform(),
        kernel_regularizer=regularizers.l2(0.01),
        bias_regularizer=regularizers.l1(0.01),
    )
    X = random.normal(key, (batch_size, input_dim))
    layer(X, training=True)
    assert len(layer.losses) == 2


def test_conv2d_layer_initializers_regularizers():
    key = random.PRNGKey(0)
    input_channels = 3
    output_channels = 8
    kernel_size = (3, 3)
    batch_size = 4
    height = 28
    width = 28

    layer = Conv2D(
        input_channels,
        output_channels,
        kernel_size,
        key,
        kernel_initializer=initializers.he_normal(),
        bias_initializer=initializers.xavier_uniform(),
        kernel_regularizer=regularizers.l2(0.01),
        bias_regularizer=regularizers.l1(0.01),
    )
    X = random.normal(key, (batch_size, height, width, input_channels))
    layer(X, training=True)
    assert len(layer.losses) == 2


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


def test_max_pooling_2d_layer():
    key = random.PRNGKey(0)
    X = random.normal(key, (1, 10, 10, 3))

    pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    output = pool(X)

    assert output.shape == (1, 5, 5, 3)


def test_avg_pooling_2d_layer():
    key = random.PRNGKey(0)
    X = random.normal(key, (1, 10, 10, 3))

    pool = AvgPooling2D(pool_size=(2, 2), strides=(2, 2))
    output = pool(X)

    assert output.shape == (1, 5, 5, 3)


def test_batch_norm_layer():
    key = random.PRNGKey(0)
    X = random.normal(key, (32, 10))

    bn = BatchNorm(10, key)
    output = bn(X, training=True)

    assert output.shape == (32, 10)
    assert not jnp.allclose(
        output.mean(axis=0), 0.0
    )  # Should not be exactly zero due to epsilon
    assert not jnp.allclose(
        output.std(axis=0), 1.0
    )  # Should not be exactly one due to epsilon


def test_layer_norm_layer():
    key = random.PRNGKey(0)
    X = random.normal(key, (32, 10, 20))

    ln = LayerNorm(shape=(10, 20))
    output = ln(X)

    assert output.shape == (32, 10, 20)
    assert jnp.allclose(output.mean(axis=(1, 2)), jnp.zeros(32), atol=1e-6)
    assert jnp.allclose(output.std(axis=(1, 2)), jnp.ones(32), atol=1e-6)
