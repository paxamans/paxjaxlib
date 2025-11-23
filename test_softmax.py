import jax.numpy as jnp

from src.paxjaxlib.activations import softmax


def test_softmax_instability():
    print("Testing softmax instability...")
    # Large values that cause exp() to overflow
    x = jnp.array([1000.0, 1001.0, 1002.0])
    print(f"Input: {x}")

    try:
        y = softmax(x)
        print(f"Output: {y}")
    except Exception as e:
        print(f"Error: {e}")

    # Check for NaNs
    if jnp.any(jnp.isnan(y)):
        print("Result contains NaNs!")
    else:
        print("Result is valid.")


if __name__ == "__main__":
    test_softmax_instability()
