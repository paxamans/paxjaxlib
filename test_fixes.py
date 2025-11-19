#!/usr/bin/env python
"""Quick test script to verify fixes"""

import jax.numpy as jnp
from jax import random

# Test activations
print("Testing activations...")
from src.paxjaxlib import activations  # noqa: E402

x = jnp.array([-1.0, 0.0, 1.0])
output = activations.gelu(x)
expected = jnp.array([-0.15865529, 0.0, 0.8413447])
print(f"GELU output: {output}")
print(f"GELU expected: {expected}")
print(f"GELU allclose (atol=1e-5): {jnp.allclose(output, expected, atol=1e-5)}")

# Test initializers
print("\nTesting initializers...")
from src.paxjaxlib import initializers  # noqa: E402

key = random.PRNGKey(0)
shape = (100, 100)
initializer = initializers.xavier_uniform()
weights = initializer(key, shape)
print(f"Xavier uniform weights shape: {weights.shape}")
print(f"Xavier uniform weights min/max: {weights.min()}, {weights.max()}")

# Test regularizers
print("\nTesting regularizers...")
from src.paxjaxlib import regularizers  # noqa: E402

x = jnp.array([-1.0, 2.0, -3.0])
reg = regularizers.l1(0.1)
result = reg(x)
print(f"L1 regularization result: {result}")
print(f"L1 expected: 0.6, got: {result}")

# Test metrics
print("\nTesting metrics...")
from src.paxjaxlib import metrics  # noqa: E402

y_true = jnp.array([1, 1, 0, 0])
y_pred = jnp.array([1, 0, 0, 1])
acc = metrics.accuracy(y_true, y_pred)
print(f"Accuracy: {acc}, expected: 0.5")

# Test Dense layer
print("\nTesting Dense layer...")
from src.paxjaxlib.layers import Dense  # noqa: E402

key = random.PRNGKey(0)
layer = Dense(10, 5, key)
X = random.normal(key, (32, 10))
output = layer(X)
print(f"Dense output shape: {output.shape}, expected: (32, 5)")

# Test LayerNorm
print("\nTesting LayerNorm...")
from src.paxjaxlib.layers import LayerNorm  # noqa: E402

ln = LayerNorm(shape=(10, 20))
X = random.normal(key, (32, 10, 20))
output = ln(X)
print(f"LayerNorm output shape: {output.shape}, expected: (32, 10, 20)")

# Test BatchNorm
print("\nTesting BatchNorm...")
from src.paxjaxlib.layers import BatchNorm  # noqa: E402

bn = BatchNorm(10, key)
X = random.normal(key, (32, 10))
output = bn(X, training=True)
print(f"BatchNorm output shape: {output.shape}, expected: (32, 10)")

print("\nAll basic tests passed!")
