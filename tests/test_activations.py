"""Tests for activation functions including new ones."""

import jax.numpy as jnp

from paxjaxlib import activations


def test_gelu():
    x = jnp.array([-1.0, 0.0, 1.0])
    output = activations.gelu(x)
    expected = jnp.array([-0.15865529, 0.0, 0.8413447])
    assert jnp.allclose(output, expected, atol=1e-6)


def test_silu():
    x = jnp.array([-1.0, 0.0, 1.0])
    output = activations.silu(x)
    expected = jnp.array([-0.26894143, 0.0, 0.7310586])
    assert jnp.allclose(output, expected)


def test_mish():
    x = jnp.array([-1.0, 0.0, 1.0])
    output = activations.mish(x)
    expected = jnp.array([-0.303373, 0.0, 0.865098])
    assert jnp.allclose(output, expected, atol=3e-5)


def test_leaky_relu_positive():
    x = jnp.array([1.0, 2.0, 3.0])
    output = activations.leaky_relu(x)
    assert jnp.allclose(output, x)


def test_leaky_relu_negative():
    x = jnp.array([-1.0, -2.0])
    output = activations.leaky_relu(x, negative_slope=0.1)
    expected = jnp.array([-0.1, -0.2])
    assert jnp.allclose(output, expected)


def test_leaky_relu_zero():
    x = jnp.array([0.0])
    output = activations.leaky_relu(x)
    assert jnp.allclose(output, jnp.array([0.0]))


def test_elu_positive():
    x = jnp.array([1.0, 2.0])
    output = activations.elu(x)
    assert jnp.allclose(output, x)


def test_elu_negative():
    x = jnp.array([-1.0])
    output = activations.elu(x, alpha=1.0)
    expected = jnp.array([jnp.exp(-1.0) - 1.0])
    assert jnp.allclose(output, expected, atol=1e-6)


def test_elu_custom_alpha():
    x = jnp.array([-1.0])
    output = activations.elu(x, alpha=2.0)
    expected = jnp.array([2.0 * (jnp.exp(-1.0) - 1.0)])
    assert jnp.allclose(output, expected, atol=1e-6)
