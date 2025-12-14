import jax.numpy as jnp

from paxjaxlib import regularizers


def test_l1():
    x = jnp.array([-1.0, 2.0, -3.0])

    reg = regularizers.l1(0.1)

    # 0.1 * (1 + 2 + 3) = 0.6
    assert jnp.isclose(reg(x), 0.6)


def test_l2():
    x = jnp.array([-1.0, 2.0, -3.0])

    reg = regularizers.l2(0.1)

    # 0.1 * (1 + 4 + 9) = 1.4
    assert jnp.isclose(reg(x), 1.4)
