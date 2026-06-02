"""
Model containers for paxjaxlib.

The :class:`NeuralNetwork` class is a simple sequential container that
chains layers together in order.  Because it inherits from
:class:`~paxjaxlib.core.Module`, the entire model is a valid JAX pytree
and can be differentiated, JIT-compiled, and vectorized.
"""

from typing import Any, List, Optional

import jax.numpy as jnp
import numpy as np
from jax import random

from .core import Module
from .layers import BatchNorm, Conv2D, Dense, Dropout, Embedding, LayerNorm


class NeuralNetwork(Module):
    """A sequential neural network.

    Layers are executed in the order they are provided.  Callable
    non-``Module`` objects (e.g. ``jax.nn.relu``) are supported and will
    be applied as plain functions.

    Args:
        layers: An ordered list of layers / callables.

    Example::

        model = NeuralNetwork([
            Dense(784, 128, key),
            jax.nn.relu,
            Dense(128, 10, key),
        ])
        y = model(x)
    """

    def __init__(self, layers: List[Module]):
        super().__init__()
        self.layers = layers
        self._built = False

    @property
    def params(self):
        """Collect parameters from all layers that have them."""
        params = []
        for layer in self.layers:
            if hasattr(layer, "params"):
                params.append(layer.params)
        return params

    def build(self, input_shape):
        if self._built:
            return
        for layer in self.layers:
            if isinstance(layer, (Dense, Conv2D, BatchNorm, LayerNorm)):
                layer.build(input_shape)
            input_shape = layer(jnp.ones(input_shape)).shape
        self._built = True

    def __call__(
        self,
        X: jnp.ndarray,
        key: Optional[Any] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        """Forward pass through the network.

        Args:
            X: Input data.
            key: Optional PRNG key (needed when ``Dropout`` layers are
                present and ``training=True``).
            training: If ``True``, enables training-time behaviour
                (dropout, batch-norm running stats, regularisation losses).
        """
        # Clear losses before forward pass
        self.clear_losses()

        current_input = X

        # If key is provided, split it for layers that need it
        iter_key = key

        for layer in self.layers:
            # Check if layer needs key/training args
            # We can check signature or just try/except, or check type.
            # Checking type is safer for our known layers.

            if isinstance(layer, Dropout):
                if training and iter_key is not None:
                    iter_key, subkey = random.split(iter_key)
                    current_input = layer(current_input, key=subkey, training=training)
                else:
                    current_input = layer(current_input, training=training)
            elif isinstance(layer, (Dense, Conv2D, BatchNorm, LayerNorm)):
                current_input = layer(current_input, training=training)
            else:
                # Other layers (Flatten, MaxPooling2D, Embedding, activations, etc.)
                current_input = layer(current_input)

        return current_input

    # ------------------------------------------------------------------
    # Serialization (numpy-based, no pickle)
    # ------------------------------------------------------------------

    def save(self, filename: str) -> None:
        """Save the model's parameters to a ``.npz`` file.

        Uses :func:`numpy.savez` instead of pickle for safety and
        cross-version compatibility.

        Args:
            filename: Destination path (e.g. ``"model.npz"``).
        """
        flat: dict[str, np.ndarray] = {}
        param_idx = 0
        for layer in self.layers:
            if hasattr(layer, "params"):
                p = layer.params
                for pname, pval in p.items():
                    flat[f"{param_idx}_{pname}"] = np.asarray(pval)
                param_idx += 1
        np.savez(filename, **flat)  # type: ignore[arg-type]

    def load(self, filename: str) -> None:
        """Load parameters from a ``.npz`` file created by :meth:`save`.

        Args:
            filename: Path to the ``.npz`` file.
        """
        # Support both .npz and legacy .pkl files
        if filename.endswith((".pkl", ".pickle", ".npy")):
            self._load_legacy_pickle(filename)
            return

        if not filename.endswith(".npz"):
            filename = filename + ".npz" if not filename.endswith(".npz") else filename

        data = np.load(filename, allow_pickle=False)
        param_idx = 0
        for layer in self.layers:
            if hasattr(layer, "params"):
                p = layer.params
                new_params = {}
                for pname in p:
                    key = f"{param_idx}_{pname}"
                    new_params[pname] = jnp.array(data[key])
                layer.params = new_params
                param_idx += 1

    def _load_legacy_pickle(self, filename: str) -> None:
        """Fallback loader for old pickle-based save files."""
        import pickle

        with open(filename, "rb") as f:
            params = pickle.load(f)
        param_idx = 0
        for layer in self.layers:
            if hasattr(layer, "params"):
                layer.params = params[param_idx]
                param_idx += 1

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary table of the model.

        Prints layer types, output shapes (where inferrable), and
        parameter counts.  Also returns the formatted string.

        Example::

            >>> model.summary()
            ┌─────┬──────────────┬────────────┐
            │   # │ Layer        │     Params │
            ├─────┼──────────────┼────────────┤
            │   0 │ Dense(10→64) │        704 │
            │   1 │ relu         │          0 │
            │   2 │ Dense(64→1)  │         65 │
            └─────┴──────────────┴────────────┘
            Total parameters: 769
        """
        rows: list[tuple[str, str, int]] = []
        total = 0
        for i, layer in enumerate(self.layers):
            name = _layer_label(layer)
            count = _param_count(layer)
            total += count
            rows.append((str(i), name, count))

        # Column widths
        w_idx = max(len(r[0]) for r in rows)
        w_name = max(len(r[1]) for r in rows)
        w_params = max(len(f"{r[2]:,}") for r in rows)

        # Ensure minimum widths for headers
        w_idx = max(w_idx, 1)
        w_name = max(w_name, 5)
        w_params = max(w_params, 6)

        def _row(idx: str, name: str, params: str) -> str:
            return f"│ {idx:>{w_idx}} │ {name:<{w_name}} │ {params:>{w_params}} │"

        top = f"┌─{'─' * w_idx}─┬─{'─' * w_name}─┬─{'─' * w_params}─┐"
        mid = f"├─{'─' * w_idx}─┼─{'─' * w_name}─┼─{'─' * w_params}─┤"
        bot = f"└─{'─' * w_idx}─┴─{'─' * w_name}─┴─{'─' * w_params}─┘"

        lines = [top, _row("#", "Layer", "Params"), mid]
        for idx_s, name, count in rows:
            lines.append(_row(idx_s, name, f"{count:,}"))
        lines.append(bot)
        lines.append(f"Total parameters: {total:,}")

        text = "\n".join(lines)
        print(text)
        return text


# ------------------------------------------------------------------
# Helpers for summary()
# ------------------------------------------------------------------


def _layer_label(layer: Any) -> str:
    """Return a short human-readable label for a layer."""
    if isinstance(layer, Dense):
        return f"Dense({layer.input_dim}→{layer.output_dim})"
    if isinstance(layer, Conv2D):
        kh, kw = layer.kernel_size
        return f"Conv2D({layer.input_channels}→{layer.output_channels}, {kh}×{kw})"
    if isinstance(layer, Dropout):
        return f"Dropout({layer.rate})"
    if isinstance(layer, Embedding):
        return f"Embedding({layer.num_embeddings}×{layer.embedding_dim})"
    if isinstance(layer, BatchNorm):
        return f"BatchNorm({layer.input_dim})"
    if isinstance(layer, LayerNorm):
        return f"LayerNorm({layer.shape})"
    if isinstance(layer, Module):
        return type(layer).__name__
    # Plain callable (e.g. jax.nn.relu)
    return getattr(layer, "__name__", type(layer).__name__)


def _param_count(layer: Any) -> int:
    """Count the number of trainable scalar parameters in a layer."""
    if not hasattr(layer, "params"):
        return 0
    total = 0
    p = layer.params
    if isinstance(p, dict):
        for v in p.values():
            if hasattr(v, "size"):
                total += int(v.size)
    return total
