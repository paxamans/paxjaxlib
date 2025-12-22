import jax.numpy as jnp

from paxjaxlib import losses


def test_huber_loss():
    y_true = jnp.array([1.0, 2.0, 3.0])
    y_pred = jnp.array([1.5, 2.5, 2.5])

    loss = losses.huber_loss(y_true, y_pred, delta=1.0)

    assert jnp.isclose(loss, 0.125)


def test_hinge_loss():
    y_true = jnp.array([1.0, -1.0, 1.0])
    y_pred = jnp.array([0.5, -0.5, 1.5])

    loss = losses.hinge_loss(y_true, y_pred)

    assert jnp.isclose(loss, 1.0 / 3)  # Mean hinge loss
