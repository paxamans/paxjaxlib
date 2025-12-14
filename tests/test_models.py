import os

import jax.numpy as jnp
import numpy as np
from jax import random

from paxjaxlib.layers import Dense
from paxjaxlib.models import NeuralNetwork


def test_model_save_and_load():
    key = random.PRNGKey(0)

    model = NeuralNetwork([Dense(10, 20, key), Dense(20, 5, key)])

    # Initialize parameters
    dummy_input = jnp.ones((1, 10))
    model(dummy_input)

    filepath = "test_model.npy"
    model.save(filepath)

    assert os.path.exists(filepath)

    new_model = NeuralNetwork([Dense(10, 20, key), Dense(20, 5, key)])
    new_model.load(filepath)

    # Check if parameters are loaded correctly
    for old_p, new_p in zip(model.params, new_model.params, strict=True):
        for key in old_p:
            assert np.allclose(old_p[key], new_p[key])

    os.remove(filepath)
