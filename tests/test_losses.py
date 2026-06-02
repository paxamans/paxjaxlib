"""Tests for loss functions including new ones."""

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


def test_cosine_similarity_loss_identical():
    """Identical vectors should give loss ≈ 0."""
    x = jnp.array([[1.0, 2.0, 3.0]])
    loss = losses.cosine_similarity_loss(x, x)
    assert jnp.isclose(loss, 0.0, atol=1e-6)


def test_cosine_similarity_loss_opposite():
    """Opposite vectors should give loss ≈ 2."""
    x = jnp.array([[1.0, 0.0, 0.0]])
    y = jnp.array([[-1.0, 0.0, 0.0]])
    loss = losses.cosine_similarity_loss(x, y)
    assert jnp.isclose(loss, 2.0, atol=1e-6)


def test_kl_divergence_identical():
    """KL divergence of a distribution with itself should be 0."""
    p = jnp.array([[0.25, 0.25, 0.25, 0.25]])
    loss = losses.kl_divergence(p, p)
    assert jnp.isclose(loss, 0.0, atol=1e-6)


def test_kl_divergence_different():
    """KL divergence of different distributions should be > 0."""
    p = jnp.array([[0.1, 0.9]])
    q = jnp.array([[0.5, 0.5]])
    loss = losses.kl_divergence(p, q)
    assert loss > 0.0
