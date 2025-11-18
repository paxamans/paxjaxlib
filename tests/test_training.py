import jax.numpy as jnp
import optax
import pytest
from jax import random

from paxjaxlib.layers import Dense
from paxjaxlib.models import NeuralNetwork
from paxjaxlib.training import Trainer


def test_trainer_step():
    key = random.PRNGKey(0)
    k1, k2 = random.split(key)
    
    model = NeuralNetwork([Dense(10, 1, k1)])
    trainer = Trainer(model, optimizer=optax.sgd(0.1), key=k2)
    
    X = random.normal(key, (32, 10))
    y = random.normal(key, (32, 1)) # Regression target
    
    initial_loss = trainer.evaluate(X, y)
    
    # Run one epoch
    trainer.train(X, y, epochs=1, batch_size=32, verbose=False)
    
    final_loss = trainer.evaluate(X, y)
    
    assert final_loss < initial_loss

def test_trainer_integration():
    # Simple XOR-like problem
    key = random.PRNGKey(42)
    X = jnp.array([[0,0], [0,1], [1,0], [1,1]], dtype=jnp.float32)
    y = jnp.array([[0], [1], [1], [0]], dtype=jnp.float32)
    
    layers = [
        Dense(2, 8, key, activation=jax.nn.relu),
        Dense(8, 1, key)
    ]
    model = NeuralNetwork(layers)
    
    trainer = Trainer(model, optimizer=optax.adam(0.05), key=key)
    
    history = trainer.train(X, y, epochs=100, batch_size=4, verbose=False)
    
    final_loss = history[-1]
    assert final_loss < 0.1
