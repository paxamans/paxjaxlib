"""Tests for NeuralNetwork model: save/load, summary."""

import os

import jax.numpy as jnp
import numpy as np
from jax import random

from paxjaxlib.layers import Dense
from paxjaxlib.models import NeuralNetwork


def test_model_save_and_load_npz():
    """Test the new numpy-based save/load."""
    key = random.PRNGKey(0)

    model = NeuralNetwork([Dense(10, 20, key), Dense(20, 5, key)])

    # Run a forward pass
    dummy_input = jnp.ones((1, 10))
    model(dummy_input)

    filepath = "test_model_save.npz"
    model.save(filepath)

    assert os.path.exists(filepath)

    new_model = NeuralNetwork([Dense(10, 20, key), Dense(20, 5, key)])
    new_model.load(filepath)

    # Check if parameters are loaded correctly
    for old_p, new_p in zip(model.params, new_model.params, strict=True):
        for k in old_p:
            assert np.allclose(old_p[k], new_p[k])

    os.remove(filepath)


def test_model_save_and_load_legacy_pickle():
    """Ensure we can still load old pickle files."""
    import pickle

    key = random.PRNGKey(0)
    model = NeuralNetwork([Dense(10, 20, key), Dense(20, 5, key)])
    dummy_input = jnp.ones((1, 10))
    model(dummy_input)

    filepath = "test_model_legacy.pkl"
    # Save as legacy pickle
    params = []
    for layer in model.layers:
        if hasattr(layer, "params"):
            params.append(layer.params)
    with open(filepath, "wb") as f:
        pickle.dump(params, f)

    new_model = NeuralNetwork([Dense(10, 20, key), Dense(20, 5, key)])
    new_model.load(filepath)

    for old_p, new_p in zip(model.params, new_model.params, strict=True):
        for k in old_p:
            assert np.allclose(old_p[k], new_p[k])

    os.remove(filepath)


def test_model_summary():
    """Test that summary() produces output and returns a string."""
    key = random.PRNGKey(0)
    model = NeuralNetwork(
        [
            Dense(10, 64, key),
            Dense(64, 1, key),
        ]
    )
    text = model.summary()
    assert isinstance(text, str)
    assert "Dense(10→64)" in text
    assert "Dense(64→1)" in text
    assert "Total parameters:" in text
