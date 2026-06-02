"""
PyTorch interoperability bridge for paxjaxlib.

Provides bidirectional conversion between paxjaxlib models and PyTorch
``nn.Sequential`` models.  **Requires** ``torch`` to be installed
(it is an optional dependency of paxjaxlib).

Quickstart::

    from paxjaxlib.torch_bridge import to_pytorch, from_pytorch

    # Export to PyTorch
    pt_model = to_pytorch(paxjaxlib_model)
    torch.save(pt_model.state_dict(), "model.pt")

    # Import from PyTorch
    jax_model = from_pytorch(pt_model, key=jax.random.PRNGKey(0))

Supported layer mappings:

    ==============================  ============================
    paxjaxlib                       PyTorch
    ==============================  ============================
    ``Dense(in, out)``              ``nn.Linear(in, out)``
    ``Conv2D(ic, oc, (kh,kw))``    ``nn.Conv2d(ic, oc, (kh,kw))``
    ``Dropout(rate)``               ``nn.Dropout(rate)``
    ``Flatten()``                   ``nn.Flatten(start_dim=1)``
    ``BatchNorm(dim)``              ``nn.BatchNorm1d(dim)``
    ``LayerNorm(shape)``            ``nn.LayerNorm(shape)``
    ``Embedding(V, D)``            ``nn.Embedding(V, D)``
    ``MaxPooling2D(ps, st)``        ``nn.MaxPool2d(ps, st)``
    ``AvgPooling2D(ps, st)``        ``nn.AvgPool2d(ps, st)``
    ``GlobalAvgPooling2D()``        ``nn.AdaptiveAvgPool2d(1) + Flatten``
    activation functions            corresponding ``nn.ReLU()`` etc.
    ==============================  ============================

Note:
    Weight transposition is handled automatically.  paxjaxlib Dense
    stores weights as ``(in, out)`` while PyTorch Linear uses ``(out, in)``.
"""

from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    raise ImportError(
        "PyTorch is required for the torch_bridge module.  "
        "Install it with: pip install paxjaxlib[torch]"
    ) from e

from . import activations
from .layers import (
    AvgPooling2D,
    BatchNorm,
    Conv2D,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    GlobalAvgPooling2D,
    LayerNorm,
    MaxPooling2D,
)
from .models import NeuralNetwork

# =====================================================================
#  Activation mapping tables
# =====================================================================

_PAXJAX_TO_TORCH_ACTIVATION: dict[Any, Callable[[], nn.Module]] = {
    activations.relu: nn.ReLU,
    activations.sigmoid: nn.Sigmoid,
    activations.tanh: nn.Tanh,
    activations.gelu: nn.GELU,
    activations.silu: nn.SiLU,
    activations.mish: nn.Mish,
    activations.softmax: lambda: nn.Softmax(dim=-1),
}

_TORCH_TO_PAXJAX_ACTIVATION = {
    nn.ReLU: activations.relu,
    nn.Sigmoid: activations.sigmoid,
    nn.Tanh: activations.tanh,
    nn.GELU: activations.gelu,
    nn.SiLU: activations.silu,
    nn.Mish: activations.mish,
    nn.Softmax: activations.softmax,
}


# =====================================================================
#  paxjaxlib → PyTorch
# =====================================================================


def to_pytorch(model: NeuralNetwork) -> nn.Sequential:
    """Convert a paxjaxlib ``NeuralNetwork`` to a PyTorch ``nn.Sequential``.

    Copies all weights into the PyTorch model.  The returned model is
    ready for inference (``model.eval()``) or fine-tuning.

    Args:
        model: A paxjaxlib :class:`NeuralNetwork`.

    Returns:
        An ``nn.Sequential`` with equivalent architecture and weights.

    Raises:
        TypeError: If a layer type is not supported for conversion.
    """
    pt_layers: list[nn.Module] = []

    for layer in model.layers:
        pt_layers.extend(_convert_layer_to_pytorch(layer))

    return nn.Sequential(*pt_layers)


def _convert_layer_to_pytorch(layer: Any) -> list[nn.Module]:
    """Convert a single paxjaxlib layer to one or more PyTorch modules."""

    # --- Dense → Linear ---
    if isinstance(layer, Dense):
        modules: list[nn.Module] = []
        in_f, out_f = layer.input_dim, layer.output_dim
        linear = nn.Linear(in_f, out_f)
        # paxjaxlib: (in, out) → PyTorch: (out, in)
        linear.weight = nn.Parameter(torch.from_numpy(np.asarray(layer.W.T)).float())
        linear.bias = nn.Parameter(torch.from_numpy(np.asarray(layer.b)).float())
        modules.append(linear)
        # Attach activation if the layer has a non-identity one
        act_mod = _activation_to_pytorch(layer.activation)
        if act_mod is not None:
            modules.append(act_mod)
        return modules

    # --- Conv2D → Conv2d ---
    if isinstance(layer, Conv2D):
        modules = []
        conv = nn.Conv2d(
            layer.input_channels,
            layer.output_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding="same" if layer.padding == "SAME" else 0,
        )
        # paxjaxlib kernel: (H, W, In, Out) → PyTorch: (Out, In, H, W)
        w_np = np.asarray(layer.W)
        conv.weight = nn.Parameter(torch.from_numpy(w_np.transpose(3, 2, 0, 1)).float())
        conv.bias = nn.Parameter(torch.from_numpy(np.asarray(layer.b)).float())
        modules.append(conv)
        act_mod = _activation_to_pytorch(layer.activation)
        if act_mod is not None:
            modules.append(act_mod)
        return modules

    # --- Dropout ---
    if isinstance(layer, Dropout):
        return [nn.Dropout(p=layer.rate)]

    # --- Flatten ---
    if isinstance(layer, Flatten):
        return [nn.Flatten(start_dim=1)]

    # --- Pooling ---
    if isinstance(layer, MaxPooling2D):
        return [nn.MaxPool2d(kernel_size=layer.pool_size, stride=layer.strides)]

    if isinstance(layer, AvgPooling2D):
        return [nn.AvgPool2d(kernel_size=layer.pool_size, stride=layer.strides)]

    if isinstance(layer, GlobalAvgPooling2D):
        return [nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim=1)]

    # --- Normalisation ---
    if isinstance(layer, BatchNorm):
        bn = nn.BatchNorm1d(
            layer.input_dim,
            eps=layer.epsilon,
            momentum=1.0 - layer.momentum,
        )
        bn.weight = nn.Parameter(torch.from_numpy(np.asarray(layer.gamma)).float())
        bn.bias = nn.Parameter(torch.from_numpy(np.asarray(layer.beta)).float())
        bn.running_mean = torch.from_numpy(np.asarray(layer.running_mean)).float()
        bn.running_var = torch.from_numpy(np.asarray(layer.running_var)).float()
        return [bn]

    if isinstance(layer, LayerNorm):
        if layer.gamma is not None:
            shape = list(layer.gamma.shape)
            ln = nn.LayerNorm(shape, eps=layer.epsilon)
            ln.weight = nn.Parameter(torch.from_numpy(np.asarray(layer.gamma)).float())
            ln.bias = nn.Parameter(torch.from_numpy(np.asarray(layer.beta)).float())
            return [ln]
        return [nn.LayerNorm(list(layer.shape or [1]), eps=layer.epsilon)]

    # --- Embedding ---
    if isinstance(layer, Embedding):
        emb = nn.Embedding(layer.num_embeddings, layer.embedding_dim)
        emb.weight = nn.Parameter(torch.from_numpy(np.asarray(layer.W)).float())
        return [emb]

    # --- Plain activation function ---
    act_mod = _activation_to_pytorch(layer)
    if act_mod is not None:
        return [act_mod]

    raise TypeError(
        f"Cannot convert layer of type {type(layer).__name__} to PyTorch. "
        f"Supported: Dense, Conv2D, Dropout, Flatten, MaxPooling2D, "
        f"AvgPooling2D, GlobalAvgPooling2D, BatchNorm, LayerNorm, "
        f"Embedding, and standard activation functions."
    )


def _activation_to_pytorch(fn: Any) -> nn.Module | None:
    """Map a paxjaxlib activation function to a PyTorch module, or None."""
    if fn is None:
        return None
    # Check identity lambda
    if callable(fn) and not isinstance(fn, type):
        try:
            # Quick check for identity lambda
            if fn.__name__ == "<lambda>":
                return None
        except AttributeError:
            pass
    for pax_fn, torch_cls in _PAXJAX_TO_TORCH_ACTIVATION.items():
        if fn is pax_fn:
            return torch_cls()
    return None


# =====================================================================
#  PyTorch → paxjaxlib
# =====================================================================


def from_pytorch(
    pt_model: nn.Sequential,
    key: Any,
) -> NeuralNetwork:
    """Convert a PyTorch ``nn.Sequential`` to a paxjaxlib ``NeuralNetwork``.

    Copies all weights from PyTorch tensors into JAX arrays.

    Args:
        pt_model: A PyTorch ``nn.Sequential``.
        key: JAX PRNG key used to initialise any layer that requires one
            (weights will be overwritten immediately, so the key value
            doesn't matter for reproducibility).

    Returns:
        A paxjaxlib :class:`NeuralNetwork` with equivalent architecture
        and weights.

    Raises:
        TypeError: If a PyTorch layer type is not supported.
    """
    from jax import random

    layers: list[Any] = []
    pt_layers = list(pt_model.children())
    i = 0
    while i < len(pt_layers):
        pt_layer = pt_layers[i]
        key, subkey = random.split(key)
        converted, consumed = _convert_layer_from_pytorch(
            pt_layer, subkey, pt_layers, i
        )
        layers.extend(converted)
        i += consumed

    return NeuralNetwork(layers)


def _convert_layer_from_pytorch(
    pt_layer: nn.Module,
    key: Any,
    all_layers: list[nn.Module],
    idx: int,
) -> tuple[list[Any], int]:
    """Convert a single PyTorch module to paxjaxlib layer(s).

    Returns (list_of_layers, num_pt_layers_consumed).
    """

    # --- Linear → Dense ---
    if isinstance(pt_layer, nn.Linear):
        in_f = pt_layer.in_features
        out_f = pt_layer.out_features
        dense = Dense(in_f, out_f, key)
        # PyTorch: (out, in) → paxjaxlib: (in, out)
        dense.W = jnp.array(pt_layer.weight.detach().numpy().T)
        if pt_layer.bias is not None:
            dense.b = jnp.array(pt_layer.bias.detach().numpy())
        else:
            dense.b = jnp.zeros((out_f,))
        # Check if next layer is an activation
        consumed = 1
        act_fn = _peek_activation_from_pytorch(all_layers, idx + 1)
        if act_fn is not None:
            dense.activation = act_fn
            consumed = 2
        return [dense], consumed

    # --- Conv2d → Conv2D ---
    if isinstance(pt_layer, nn.Conv2d):
        ic = pt_layer.in_channels
        oc = pt_layer.out_channels
        conv_ks = pt_layer.kernel_size
        conv_stride = pt_layer.stride
        pad = "SAME" if pt_layer.padding == "same" else "VALID"
        ks_pair = (
            (conv_ks[0], conv_ks[1])
            if isinstance(conv_ks, (tuple, list))
            else (conv_ks, conv_ks)
        )
        stride_pair = (
            (conv_stride[0], conv_stride[1])
            if isinstance(conv_stride, (tuple, list))
            else (conv_stride, conv_stride)
        )
        conv = Conv2D(ic, oc, ks_pair, key, stride=stride_pair, padding=pad)
        # PyTorch: (Out, In, H, W) → paxjaxlib: (H, W, In, Out)
        w_np = pt_layer.weight.detach().numpy()
        conv.W = jnp.array(w_np.transpose(2, 3, 1, 0))
        if pt_layer.bias is not None:
            conv.b = jnp.array(pt_layer.bias.detach().numpy())
        else:
            conv.b = jnp.zeros((oc,))
        consumed = 1
        act_fn = _peek_activation_from_pytorch(all_layers, idx + 1)
        if act_fn is not None:
            conv.activation = act_fn
            consumed = 2
        return [conv], consumed

    # --- Dropout ---
    if isinstance(pt_layer, nn.Dropout):
        return [Dropout(rate=pt_layer.p)], 1

    # --- Flatten ---
    if isinstance(pt_layer, nn.Flatten):
        return [Flatten()], 1

    # --- Pooling ---
    if isinstance(pt_layer, nn.MaxPool2d):
        maxpool_ks = pt_layer.kernel_size
        maxpool_stride = pt_layer.stride
        maxpool_ks_pair = (
            (maxpool_ks, maxpool_ks)
            if isinstance(maxpool_ks, int)
            else (maxpool_ks[0], maxpool_ks[1])
        )
        maxpool_st_pair = (
            (maxpool_stride, maxpool_stride)
            if isinstance(maxpool_stride, int)
            else (maxpool_stride[0], maxpool_stride[1])
        )
        return [MaxPooling2D(pool_size=maxpool_ks_pair, strides=maxpool_st_pair)], 1

    if isinstance(pt_layer, nn.AvgPool2d):
        avgpool_ks = pt_layer.kernel_size
        avgpool_stride = pt_layer.stride
        avgpool_ks_pair = (
            (avgpool_ks, avgpool_ks)
            if isinstance(avgpool_ks, int)
            else (avgpool_ks[0], avgpool_ks[1])
        )
        avgpool_st_pair = (
            (avgpool_stride, avgpool_stride)
            if isinstance(avgpool_stride, int)
            else (avgpool_stride[0], avgpool_stride[1])
        )
        return [AvgPooling2D(pool_size=avgpool_ks_pair, strides=avgpool_st_pair)], 1

    if isinstance(pt_layer, nn.AdaptiveAvgPool2d):
        return [GlobalAvgPooling2D()], 1

    # --- Normalisation ---
    if isinstance(pt_layer, nn.BatchNorm1d):
        bn = BatchNorm(
            pt_layer.num_features,
            key,
            momentum=1.0 - pt_layer.momentum,
            epsilon=pt_layer.eps,
        )
        if pt_layer.weight is not None:
            bn.gamma = jnp.array(pt_layer.weight.detach().numpy())
        if pt_layer.bias is not None:
            bn.beta = jnp.array(pt_layer.bias.detach().numpy())
        if pt_layer.running_mean is not None:
            bn.running_mean = jnp.array(pt_layer.running_mean.detach().numpy())
        if pt_layer.running_var is not None:
            bn.running_var = jnp.array(pt_layer.running_var.detach().numpy())
        return [bn], 1

    if isinstance(pt_layer, nn.LayerNorm):
        shape = tuple(pt_layer.normalized_shape)
        ln = LayerNorm(shape=shape, epsilon=pt_layer.eps)
        if pt_layer.weight is not None:
            ln.gamma = jnp.array(pt_layer.weight.detach().numpy())
        if pt_layer.bias is not None:
            ln.beta = jnp.array(pt_layer.bias.detach().numpy())
        return [ln], 1

    # --- Embedding ---
    if isinstance(pt_layer, nn.Embedding):
        emb = Embedding(pt_layer.num_embeddings, pt_layer.embedding_dim, key)
        emb.W = jnp.array(pt_layer.weight.detach().numpy())
        return [emb], 1

    # --- Standalone activation ---
    act_fn = _torch_module_to_activation(pt_layer)
    if act_fn is not None:
        return [act_fn], 1

    raise TypeError(
        f"Cannot convert PyTorch module of type {type(pt_layer).__name__} "
        f"to paxjaxlib."
    )


def _peek_activation_from_pytorch(layers: list[nn.Module], idx: int) -> Any | None:
    """If layers[idx] is a PyTorch activation module, return the
    corresponding paxjaxlib function. Otherwise return None."""
    if idx >= len(layers):
        return None
    return _torch_module_to_activation(layers[idx])


def _torch_module_to_activation(module: nn.Module) -> Any | None:
    """Map a PyTorch activation module to a paxjaxlib function."""
    for torch_cls, pax_fn in _TORCH_TO_PAXJAX_ACTIVATION.items():
        if isinstance(module, torch_cls):
            return pax_fn
    return None


# =====================================================================
#  Convenience I/O
# =====================================================================


def save_as_pytorch(model: NeuralNetwork, path: str) -> None:
    """Convert a paxjaxlib model to PyTorch and save it.

    The saved file contains the PyTorch ``state_dict`` and can be loaded
    with standard ``torch.load()``.

    Args:
        model: paxjaxlib model to export.
        path: Destination file path (e.g. ``"model.pt"``).
    """
    pt_model = to_pytorch(model)
    torch.save(pt_model.state_dict(), path)


def load_from_pytorch(
    path: str,
    key: Any,
    pt_model: nn.Sequential | None = None,
) -> NeuralNetwork:
    """Load a PyTorch model file and convert to paxjaxlib.

    Args:
        path: Path to a saved PyTorch model (full model, not just
            state_dict).
        key: JAX PRNG key for layer construction.
        pt_model: If provided, the state_dict from ``path`` is loaded
            into this model before conversion.  This is needed when the
            file contains only a state_dict.

    Returns:
        A paxjaxlib :class:`NeuralNetwork`.
    """
    if pt_model is not None:
        state_dict = torch.load(path, weights_only=True)
        pt_model.load_state_dict(state_dict)
        return from_pytorch(pt_model, key)
    else:
        loaded = torch.load(path, weights_only=False)
        if isinstance(loaded, nn.Sequential):
            return from_pytorch(loaded, key)
        raise TypeError(
            "The file does not contain an nn.Sequential model. "
            "Pass a `pt_model` argument to load a state_dict instead."
        )
