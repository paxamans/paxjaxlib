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
