import jax.numpy as jnp

from src.paxjaxlib import activations

x = jnp.array([-1.0, 0.0, 1.0])
gelu_out = activations.gelu(x)
mish_out = activations.mish(x)

print(f"GELU: {gelu_out}")
print(f"MISH: {mish_out}")

expected_gelu = jnp.array([-0.15865529, 0.0, 0.8413447])
expected_mish = jnp.array([-0.303373, 0.0, 0.865098])

print(f"GELU Diff: {jnp.abs(gelu_out - expected_gelu)}")
print(f"MISH Diff: {jnp.abs(mish_out - expected_mish)}")
