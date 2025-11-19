#!/usr/bin/env python
"""Verify specific test cases"""

import sys

import jax.numpy as jnp
from jax import random

# Add src to path
sys.path.insert(0, "/c/Users/lanyp/github/paxjaxlib")

print("=" * 60)
print("Test 1: GELU activation precision")
print("=" * 60)

from src.paxjaxlib import activations  # noqa: E402

x = jnp.array([-1.0, 0.0, 1.0])
output = activations.gelu(x)
expected = jnp.array([-0.15865529, 0.0, 0.8413447])

print(f"Input: {x}")
print(f"Output:   {output}")
print(f"Expected: {expected}")
print(f"Difference: {jnp.abs(output - expected)}")
print(f"allclose(atol=1e-6): {jnp.allclose(output, expected, atol=1e-6)}")
print(f"allclose(atol=1e-5): {jnp.allclose(output, expected, atol=1e-5)}")
print(f"allclose(atol=1e-4): {jnp.allclose(output, expected, atol=1e-4)}")

print("\n" + "=" * 60)
print("Test 2: MISH activation precision")
print("=" * 60)

output_mish = activations.mish(x)
expected_mish = jnp.array([-0.303373, 0.0, 0.865098])

print(f"Input: {x}")
print(f"Output:   {output_mish}")
print(f"Expected: {expected_mish}")
print(f"Difference: {jnp.abs(output_mish - expected_mish)}")
print(f"allclose(atol=1e-6): {jnp.allclose(output_mish, expected_mish, atol=1e-6)}")
print(f"allclose(atol=1e-5): {jnp.allclose(output_mish, expected_mish, atol=1e-5)}")
print(f"allclose(atol=1e-4): {jnp.allclose(output_mish, expected_mish, atol=1e-4)}")

print("\n" + "=" * 60)
print("Test 3: Initializers return callables")
print("=" * 60)

from src.paxjaxlib import initializers  # noqa: E402

key = random.PRNGKey(0)
xavier_init = initializers.xavier_uniform()
print(f"xavier_uniform() type: {type(xavier_init)}")
print(f"Callable: {callable(xavier_init)}")

weights = xavier_init(key, (100, 100))
print(f"Generated weights shape: {weights.shape}")

print("\n" + "=" * 60)
print("Test 4: Regularizers return callables")
print("=" * 60)

from src.paxjaxlib import regularizers  # noqa: E402

x = jnp.array([-1.0, 2.0, -3.0])
l1_reg = regularizers.l1(0.1)
print(f"l1(0.1) type: {type(l1_reg)}")
print(f"Callable: {callable(l1_reg)}")

result = l1_reg(x)
print("L1 regularization result: {result}")
print("Expected: 0.6 (0.1 * (1 + 2 + 3))")
print(f"isclose: {jnp.isclose(result, 0.6)}")

print("\n" + "=" * 60)
print("Test 5: Dense layer with string activation")
print("=" * 60)

from src.paxjaxlib.layers import Dense  # noqa: E402

key = random.PRNGKey(0)
try:
    layer = Dense(2, 8, key, activation="relu")
    print("Dense layer created successfully with activation='relu'")
    print(f"Activation function: {layer.activation}")

    X = jnp.ones((4, 2))
    output = layer(X)
    print(f"Output shape: {output.shape}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("Test 6: Metrics argument order")
print("=" * 60)

from src.paxjaxlib import metrics  # noqa: E402

y_true = jnp.array([1, 1, 0, 0])
y_pred = jnp.array([1, 0, 0, 1])

acc = metrics.accuracy(y_true, y_pred)
print(f"accuracy(y_true, y_pred): {acc}")
print("Expected: 0.5")
print(f"Match: {jnp.isclose(acc, 0.5)}")

print("\nAll verification tests completed!")
