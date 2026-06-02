"""Tests for the PyTorch bridge (paxjaxlib ↔ PyTorch conversion).

All tests are skipped if torch is not installed.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from paxjaxlib import activations
from paxjaxlib.layers import (
    Dense,
    Dropout,
    Embedding,
    Flatten,
)
from paxjaxlib.models import NeuralNetwork

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


@pytest.fixture
def key():
    return random.PRNGKey(42)


class TestToPytorch:
    """paxjaxlib → PyTorch conversion."""

    def test_dense_roundtrip_output(self, key):
        from paxjaxlib.torch_bridge import to_pytorch

        model = NeuralNetwork([Dense(4, 8, key)])
        pt_model = to_pytorch(model)

        x_np = np.random.randn(2, 4).astype(np.float32)
        jax_out = np.asarray(model(jnp.array(x_np)))

        pt_model.eval()
        with torch.no_grad():
            pt_out = pt_model(torch.from_numpy(x_np)).numpy()

        np.testing.assert_allclose(jax_out, pt_out, atol=1e-5)

    def test_dense_with_activation(self, key):
        from paxjaxlib.torch_bridge import to_pytorch

        model = NeuralNetwork([Dense(4, 8, key, activation=activations.relu)])
        pt_model = to_pytorch(model)

        x_np = np.random.randn(2, 4).astype(np.float32)
        jax_out = np.asarray(model(jnp.array(x_np)))

        pt_model.eval()
        with torch.no_grad():
            pt_out = pt_model(torch.from_numpy(x_np)).numpy()

        np.testing.assert_allclose(jax_out, pt_out, atol=1e-5)

    def test_dropout_converted(self, key):
        from paxjaxlib.torch_bridge import to_pytorch

        model = NeuralNetwork([Dense(4, 4, key), Dropout(0.5)])
        pt_model = to_pytorch(model)

        # Check that a Dropout module exists in the PT model
        has_dropout = any(isinstance(m, nn.Dropout) for m in pt_model.modules())
        assert has_dropout

    def test_flatten_converted(self, key):
        from paxjaxlib.torch_bridge import to_pytorch

        model = NeuralNetwork([Flatten()])
        pt_model = to_pytorch(model)
        has_flatten = any(isinstance(m, nn.Flatten) for m in pt_model.modules())
        assert has_flatten

    def test_embedding_roundtrip(self, key):
        from paxjaxlib.torch_bridge import to_pytorch

        model = NeuralNetwork([Embedding(50, 16, key)])
        pt_model = to_pytorch(model)

        ids_np = np.array([[1, 2, 3]])
        jax_out = np.asarray(model(jnp.array(ids_np)))

        pt_model.eval()
        with torch.no_grad():
            pt_out = pt_model(torch.from_numpy(ids_np).long()).numpy()

        np.testing.assert_allclose(jax_out, pt_out, atol=1e-5)

    def test_multi_layer_sequential(self, key):
        from paxjaxlib.torch_bridge import to_pytorch

        k1, k2, k3 = random.split(key, 3)
        model = NeuralNetwork(
            [
                Dense(10, 32, k1, activation=activations.relu),
                Dense(32, 16, k2, activation=activations.sigmoid),
                Dense(16, 1, k3),
            ]
        )
        pt_model = to_pytorch(model)

        x_np = np.random.randn(4, 10).astype(np.float32)
        jax_out = np.asarray(model(jnp.array(x_np)))

        pt_model.eval()
        with torch.no_grad():
            pt_out = pt_model(torch.from_numpy(x_np)).numpy()

        np.testing.assert_allclose(jax_out, pt_out, atol=1e-5)


class TestFromPytorch:
    """PyTorch → paxjaxlib conversion."""

    def test_linear_roundtrip(self, key):
        from paxjaxlib.torch_bridge import from_pytorch

        pt_model = nn.Sequential(nn.Linear(4, 8))
        jax_model = from_pytorch(pt_model, key)

        x_np = np.random.randn(2, 4).astype(np.float32)

        pt_model.eval()
        with torch.no_grad():
            pt_out = pt_model(torch.from_numpy(x_np)).numpy()

        jax_out = np.asarray(jax_model(jnp.array(x_np)))
        np.testing.assert_allclose(jax_out, pt_out, atol=1e-5)

    def test_linear_with_relu(self, key):
        from paxjaxlib.torch_bridge import from_pytorch

        pt_model = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        jax_model = from_pytorch(pt_model, key)

        x_np = np.random.randn(2, 4).astype(np.float32)

        pt_model.eval()
        with torch.no_grad():
            pt_out = pt_model(torch.from_numpy(x_np)).numpy()

        jax_out = np.asarray(jax_model(jnp.array(x_np)))
        np.testing.assert_allclose(jax_out, pt_out, atol=1e-5)

    def test_dropout_imported(self, key):
        from paxjaxlib.torch_bridge import from_pytorch

        pt_model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.3))
        jax_model = from_pytorch(pt_model, key)

        has_dropout = any(isinstance(layer, Dropout) for layer in jax_model.layers)
        assert has_dropout

    def test_embedding_imported(self, key):
        from paxjaxlib.torch_bridge import from_pytorch

        pt_emb = nn.Embedding(100, 32)
        pt_model = nn.Sequential(pt_emb)
        jax_model = from_pytorch(pt_model, key)

        has_emb = any(isinstance(layer, Embedding) for layer in jax_model.layers)
        assert has_emb


class TestFullRoundTrip:
    """paxjaxlib → PyTorch → paxjaxlib → verify identical outputs."""

    def test_dense_full_roundtrip(self, key):
        from paxjaxlib.torch_bridge import from_pytorch, to_pytorch

        k1, k2 = random.split(key)
        original = NeuralNetwork(
            [
                Dense(8, 16, k1, activation=activations.relu),
                Dense(16, 4, k2),
            ]
        )

        x_np = np.random.randn(3, 8).astype(np.float32)
        original_out = np.asarray(original(jnp.array(x_np)))

        # paxjaxlib → PyTorch → paxjaxlib
        pt_model = to_pytorch(original)
        recovered = from_pytorch(pt_model, key)
        recovered_out = np.asarray(recovered(jnp.array(x_np)))

        np.testing.assert_allclose(original_out, recovered_out, atol=1e-5)
