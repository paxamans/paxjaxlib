"""Tests for new layers: Embedding, GlobalAvgPooling2D, MultiHeadAttention."""

import jax
import jax.numpy as jnp
from jax import random

from paxjaxlib.layers import Embedding, GlobalAvgPooling2D, MultiHeadAttention


class TestEmbedding:
    def test_output_shape(self):
        key = random.PRNGKey(0)
        emb = Embedding(num_embeddings=100, embedding_dim=32, key=key)
        ids = jnp.array([[1, 5, 10], [2, 3, 7]])  # (2, 3)
        out = emb(ids)
        assert out.shape == (2, 3, 32)

    def test_single_dim_input(self):
        key = random.PRNGKey(0)
        emb = Embedding(num_embeddings=50, embedding_dim=16, key=key)
        ids = jnp.array([0, 1, 2])
        out = emb(ids)
        assert out.shape == (3, 16)

    def test_gradient_flows(self):
        key = random.PRNGKey(0)
        emb = Embedding(num_embeddings=10, embedding_dim=8, key=key)

        def loss_fn(emb_layer):
            ids = jnp.array([0, 1, 2])
            return jnp.sum(emb_layer(ids))

        grads = jax.grad(loss_fn)(emb)
        assert grads.W.shape == (10, 8)

    def test_params_property(self):
        key = random.PRNGKey(0)
        emb = Embedding(num_embeddings=10, embedding_dim=4, key=key)
        p = emb.params
        assert "W" in p
        assert p["W"].shape == (10, 4)


class TestGlobalAvgPooling2D:
    def test_output_shape(self):
        pool = GlobalAvgPooling2D()
        X = jnp.ones((4, 8, 8, 16))  # (N, H, W, C)
        out = pool(X)
        assert out.shape == (4, 16)

    def test_correct_mean(self):
        pool = GlobalAvgPooling2D()
        key = random.PRNGKey(42)
        X = random.normal(key, (2, 4, 4, 3))
        out = pool(X)
        expected = jnp.mean(X, axis=(1, 2))
        assert jnp.allclose(out, expected)


class TestMultiHeadAttention:
    def test_output_shape(self):
        key = random.PRNGKey(0)
        attn = MultiHeadAttention(embed_dim=64, num_heads=8, key=key)
        X = random.normal(key, (2, 10, 64))  # (batch, seq_len, embed_dim)
        out = attn(X)
        assert out.shape == (2, 10, 64)

    def test_single_head(self):
        key = random.PRNGKey(1)
        attn = MultiHeadAttention(embed_dim=32, num_heads=1, key=key)
        X = random.normal(key, (1, 5, 32))
        out = attn(X)
        assert out.shape == (1, 5, 32)

    def test_with_mask(self):
        key = random.PRNGKey(2)
        attn = MultiHeadAttention(embed_dim=16, num_heads=2, key=key)
        X = random.normal(key, (1, 4, 16))
        # Causal mask: True = masked
        mask = jnp.triu(jnp.ones((1, 1, 4, 4), dtype=bool), k=1)
        out = attn(X, mask=mask)
        assert out.shape == (1, 4, 16)

    def test_gradient_flows(self):
        key = random.PRNGKey(3)
        attn = MultiHeadAttention(embed_dim=16, num_heads=2, key=key)

        def loss_fn(attn_layer):
            X = jnp.ones((1, 3, 16))
            return jnp.sum(attn_layer(X))

        grads = jax.grad(loss_fn)(attn)
        assert grads.W_q.shape == (16, 16)

    def test_params_property(self):
        key = random.PRNGKey(0)
        attn = MultiHeadAttention(embed_dim=32, num_heads=4, key=key)
        p = attn.params
        assert set(p.keys()) == {"W_q", "W_k", "W_v", "W_o"}

    def test_invalid_dims_raises(self):
        import pytest

        key = random.PRNGKey(0)
        with pytest.raises(ValueError):
            MultiHeadAttention(embed_dim=10, num_heads=3, key=key)
