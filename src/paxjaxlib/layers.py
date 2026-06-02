"""
Neural network layers for paxjaxlib.

Every layer inherits from :class:`~paxjaxlib.core.Module` which
automatically registers it as a JAX pytree.  Layers are designed to be
composed inside a :class:`~paxjaxlib.models.NeuralNetwork`.
"""

from typing import Any, Callable, Optional, Tuple, Union, cast

import jax.lax as lax
import jax.numpy as jnp
from jax import random

from . import activations
from .core import Module
from .initializers import he_normal

# =====================================================================
# Regularisation / Utility layers
# =====================================================================


class Dropout(Module):
    """Randomly zeroes elements during training (inverted dropout).

    Args:
        rate: Fraction of input units to drop (must be in ``[0, 1)``).
    """

    def __init__(self, rate: float):
        super().__init__()
        if not (0.0 <= rate < 1.0):
            raise ValueError("Dropout rate must be in the interval [0, 1).")
        self.rate = rate

    def __call__(
        self,
        X: jnp.ndarray,
        key: Optional[Any] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        """Apply dropout.

        Args:
            X: Input data.
            key: JAX PRNGKey for dropout. Required if training is True.
            training: If True, applies dropout.
        """
        if not training or self.rate <= 0.0:
            return X

        if key is None:
            raise ValueError("Dropout layer requires a PRNGKey during training.")

        keep_prob = 1.0 - self.rate
        mask = random.bernoulli(key, p=keep_prob, shape=X.shape)
        return (X * mask) / keep_prob


# =====================================================================
# Convolutional layers
# =====================================================================


class Conv2D(Module):
    """2-D convolution layer.

    Operates on inputs in NHWC format.

    Args:
        input_channels: Number of channels in the input.
        output_channels: Number of filters (output channels).
        kernel_size: ``(height, width)`` of each filter.
        key: PRNG key for weight initialisation.
        activation: Activation function applied after the convolution.
        stride: ``(h_stride, w_stride)``. Default ``(1, 1)``.
        padding: ``"SAME"`` or ``"VALID"``. Default ``"SAME"``.
        kernel_initializer: Weight initialiser. Default :func:`he_normal`.
        bias_initializer: Bias initialiser. Default zeros.
        kernel_regularizer: Optional regulariser for the kernel.
        bias_regularizer: Optional regulariser for the bias.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: Tuple[int, int],
        key: Any,
        activation: Callable = lambda x: x,
        stride: Tuple[int, int] = (1, 1),
        padding: str = "SAME",
        kernel_initializer: Optional[Callable] = None,
        bias_initializer: Optional[Callable] = None,
        kernel_regularizer: Optional[Callable] = None,
        bias_regularizer: Optional[Callable] = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.stride = stride
        self.padding = padding
        self.kernel_initializer = kernel_initializer or he_normal()
        self.bias_initializer = bias_initializer or (
            lambda key, shape: jnp.zeros(shape)
        )
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        # Initialize weights and biases
        self.W = self.kernel_initializer(
            key,
            (
                self.kernel_size[0],
                self.kernel_size[1],
                self.input_channels,
                self.output_channels,
            ),
        )
        self.b = self.bias_initializer(key, (self.output_channels,))

    @property
    def params(self):
        return {"W": self.W, "b": self.b}

    @params.setter
    def params(self, value):
        if isinstance(value, dict):
            self.W = value["W"]
            self.b = value["b"]

    def __call__(self, X: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if training:
            if self.kernel_regularizer:
                self.add_loss(self.kernel_regularizer(self.W))
            if self.bias_regularizer:
                self.add_loss(self.bias_regularizer(self.b))
        conv_output = lax.conv_general_dilated(
            X,
            self.W,
            window_strides=self.stride,
            padding=self.padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=1,
        )
        return cast(
            jnp.ndarray, self.activation(conv_output + self.b[None, None, None, :])
        )


# =====================================================================
# Reshape / Pooling layers
# =====================================================================


class Flatten(Module):
    """Flattens all dimensions except the batch dimension.

    ``(N, H, W, C) → (N, H*W*C)``
    """

    def __init__(self):
        super().__init__()

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return X.reshape(X.shape[0], -1)


class MaxPooling2D(Module):
    """2-D max pooling over spatial dimensions (NHWC format).

    Args:
        pool_size: ``(height, width)`` of the pooling window.
        strides: ``(h_stride, w_stride)``. Defaults to ``pool_size``.
        padding: ``"VALID"`` (default) or ``"SAME"``.
    """

    def __init__(
        self,
        pool_size: Tuple[int, int],
        strides: Optional[Tuple[int, int]] = None,
        padding: str = "VALID",
    ):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides or pool_size
        self.padding = padding

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return cast(
            jnp.ndarray,
            lax.reduce_window(
                X,
                -jnp.inf,
                lax.max,
                (1, *self.pool_size, 1),
                (1, *self.strides, 1),
                self.padding,
            ),
        )


class AvgPooling2D(Module):
    """2-D average pooling over spatial dimensions (NHWC format).

    Args:
        pool_size: ``(height, width)`` of the pooling window.
        strides: ``(h_stride, w_stride)``. Defaults to ``pool_size``.
        padding: ``"VALID"`` (default) or ``"SAME"``.
    """

    def __init__(
        self,
        pool_size: Tuple[int, int],
        strides: Optional[Tuple[int, int]] = None,
        padding: str = "VALID",
    ):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides or pool_size
        self.padding = padding

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        counts = lax.reduce_window(
            jnp.ones_like(X),
            0.0,
            lax.add,
            (1, *self.pool_size, 1),
            (1, *self.strides, 1),
            self.padding,
        )
        sums = lax.reduce_window(
            X,
            0.0,
            lax.add,
            (1, *self.pool_size, 1),
            (1, *self.strides, 1),
            self.padding,
        )
        return cast(jnp.ndarray, sums / counts)


class GlobalAvgPooling2D(Module):
    """Global average pooling over spatial dimensions.

    Reduces ``(N, H, W, C) → (N, C)`` by averaging over H and W.
    Commonly used right before a final Dense classification head.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(X, axis=(1, 2))


# =====================================================================
# Normalisation layers
# =====================================================================


class BatchNorm(Module):
    """Batch Normalisation layer.

    Normalises each feature across the batch dimension during training,
    and uses running statistics at inference time.

    Args:
        input_dim: Number of features.
        key: PRNG key (unused, kept for API consistency).
        momentum: Running-stats momentum. Default ``0.99``.
        epsilon: Small constant for numerical stability. Default ``1e-5``.
    """

    def __init__(self, input_dim: int, key: Any, momentum=0.99, epsilon=1e-5):
        super().__init__()
        self.input_dim = input_dim
        self.momentum = momentum
        self.epsilon = epsilon
        self.gamma = jnp.ones(input_dim)
        self.beta = jnp.zeros(input_dim)
        self.running_mean = jnp.zeros(input_dim)
        self.running_var = jnp.ones(input_dim)

    @property
    def params(self):
        return {
            "gamma": self.gamma,
            "beta": self.beta,
            "running_mean": self.running_mean,
            "running_var": self.running_var,
        }

    @params.setter
    def params(self, value):
        if isinstance(value, dict):
            self.gamma = value["gamma"]
            self.beta = value["beta"]
            self.running_mean = value["running_mean"]
            self.running_var = value["running_var"]

    def __call__(self, X: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if training:
            mean = jnp.mean(X, axis=0)
            var = jnp.var(X, axis=0)
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )
        else:
            mean = self.running_mean
            var = self.running_var
        return self.gamma * (X - mean) / jnp.sqrt(var + self.epsilon) + self.beta


class LayerNorm(Module):
    """Layer Normalisation.

    Normalises each sample across the feature dimension(s).

    Args:
        shape: Shape of the learnable affine parameters.  If ``None``,
            parameters are lazily initialised on the first call.
        epsilon: Small constant for numerical stability. Default ``1e-5``.
    """

    def __init__(self, shape: Optional[Tuple] = None, epsilon=1e-5):
        super().__init__()
        self.shape = shape
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None
        if shape is not None:
            self.gamma = jnp.ones(shape)
            self.beta = jnp.zeros(shape)

    @property
    def params(self):
        return {"gamma": self.gamma, "beta": self.beta}

    @params.setter
    def params(self, value):
        if isinstance(value, dict):
            self.gamma = value["gamma"]
            self.beta = value["beta"]

    def build(self, input_shape):
        if self.gamma is None:
            self.gamma = jnp.ones(input_shape[-1])
            self.beta = jnp.zeros(input_shape[-1])

    def __call__(self, X: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if self.gamma is None:
            # Initialize on first call if not already initialized
            if self.shape is None:
                self.gamma = jnp.ones(X.shape[-1:])
                self.beta = jnp.zeros(X.shape[-1:])
        mean = jnp.mean(X, axis=-1, keepdims=True)
        var = jnp.var(X, axis=-1, keepdims=True)
        if self.gamma is None or self.beta is None:
            raise ValueError("LayerNorm not initialized")
        return self.gamma * (X - mean) / jnp.sqrt(var + self.epsilon) + self.beta


# =====================================================================
# Dense / Linear layers
# =====================================================================


class Dense(Module):
    """Fully-connected (dense) layer.

    Computes ``activation(X @ W + b)``.

    Args:
        input_dim: Size of the input feature dimension.
        output_dim: Size of the output feature dimension.
        key: PRNG key for weight initialisation.
        activation: Activation function or its string name (e.g. ``"relu"``).
        kernel_initializer: Weight initialiser. Default :func:`he_normal`.
        bias_initializer: Bias initialiser. Default zeros.
        kernel_regularizer: Optional regulariser for the kernel.
        bias_regularizer: Optional regulariser for the bias.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        key: Any,
        activation: Union[Callable, str, None] = None,
        kernel_initializer: Optional[Callable] = None,
        bias_initializer: Optional[Callable] = None,
        kernel_regularizer: Optional[Callable] = None,
        bias_regularizer: Optional[Callable] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Handle string activations
        if isinstance(activation, str):
            self.activation = getattr(activations, activation.lower())
        else:
            self.activation = activation if activation is not None else lambda x: x
        self.kernel_initializer = kernel_initializer or he_normal()
        self.bias_initializer = bias_initializer or (
            lambda key, shape: jnp.zeros(shape)
        )
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        # Initialize weights and biases immediately
        self.W = self.kernel_initializer(key, (self.input_dim, self.output_dim))
        self.b = self.bias_initializer(key, (self.output_dim,))
        self.built = True

    @property
    def params(self):
        """Return parameters as a dictionary."""
        return {"W": self.W, "b": self.b}

    @params.setter
    def params(self, value):
        """Set parameters from a dictionary."""
        if isinstance(value, dict):
            self.W = value["W"]
            self.b = value["b"]

    def __call__(self, X: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if training:
            if self.kernel_regularizer:
                self.add_loss(self.kernel_regularizer(self.W))
            if self.bias_regularizer:
                self.add_loss(self.bias_regularizer(self.b))
        Z = jnp.dot(X, self.W) + self.b
        if self.activation is not None:
            return cast(jnp.ndarray, self.activation(Z))
        return Z


# =====================================================================
# Embedding layers
# =====================================================================


class Embedding(Module):
    """Lookup-table embedding layer.

    Maps integer token indices to dense vectors.  This is the standard
    first layer in NLP / sequence models.

    Args:
        num_embeddings: Size of the vocabulary (number of distinct tokens).
        embedding_dim: Dimensionality of each embedding vector.
        key: PRNG key for weight initialisation.

    Example::

        emb = Embedding(vocab_size=10_000, embedding_dim=128, key=key)
        vectors = emb(token_ids)  # (batch, seq_len) → (batch, seq_len, 128)
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, key: Any):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.W = random.normal(key, (num_embeddings, embedding_dim)) * 0.01

    @property
    def params(self):
        """Return the embedding matrix."""
        return {"W": self.W}

    @params.setter
    def params(self, value):
        if isinstance(value, dict):
            self.W = value["W"]

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """Look up embeddings for integer indices.

        Args:
            X: Integer array of token indices, shape ``(batch, seq_len)``
                or ``(batch,)``.

        Returns:
            Embedding vectors with an extra trailing dimension.
        """
        return self.W[X]


# =====================================================================
# Attention layers
# =====================================================================


class MultiHeadAttention(Module):
    """Multi-head self-attention (as in *Attention Is All You Need*).

    Applies scaled dot-product attention with ``num_heads`` parallel
    heads, each operating on ``embed_dim // num_heads`` dimensions.

    Args:
        embed_dim: Total embedding / model dimension.
        num_heads: Number of attention heads.  Must evenly divide
            ``embed_dim``.
        key: PRNG key for weight initialisation.

    Example::

        attn = MultiHeadAttention(embed_dim=256, num_heads=8, key=key)
        out = attn(x)  # (batch, seq_len, 256) → (batch, seq_len, 256)
    """

    def __init__(self, embed_dim: int, num_heads: int, key: Any):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Split the master key for each projection matrix
        k1, k2, k3, k4 = random.split(key, 4)
        scale = 1.0 / jnp.sqrt(float(self.head_dim))
        self.W_q = random.normal(k1, (embed_dim, embed_dim)) * scale
        self.W_k = random.normal(k2, (embed_dim, embed_dim)) * scale
        self.W_v = random.normal(k3, (embed_dim, embed_dim)) * scale
        self.W_o = random.normal(k4, (embed_dim, embed_dim)) * scale

    @property
    def params(self):
        """Return projection matrices as a dict."""
        return {"W_q": self.W_q, "W_k": self.W_k, "W_v": self.W_v, "W_o": self.W_o}

    @params.setter
    def params(self, value):
        if isinstance(value, dict):
            self.W_q = value["W_q"]
            self.W_k = value["W_k"]
            self.W_v = value["W_v"]
            self.W_o = value["W_o"]

    def __call__(
        self,
        X: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Compute multi-head self-attention.

        Args:
            X: Input of shape ``(batch, seq_len, embed_dim)``.
            mask: Optional boolean mask of shape ``(batch, 1, 1, seq_len)``
                or ``(batch, 1, seq_len, seq_len)``.  ``True`` values are
                **masked out** (set to ``-inf`` before softmax).

        Returns:
            Output of the same shape as *X*.
        """
        batch, seq_len, _ = X.shape

        # Linear projections
        Q = X @ self.W_q  # (B, S, D)
        K = X @ self.W_k
        V = X @ self.W_v

        # Reshape into (B, num_heads, S, head_dim)
        Q = Q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        K = K.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        V = V.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        # Scaled dot-product attention
        scale = jnp.sqrt(float(self.head_dim))
        attn_weights = (Q @ K.transpose(0, 1, 3, 2)) / scale  # (B, H, S, S)

        if mask is not None:
            attn_weights = jnp.where(mask, -1e9, attn_weights)

        attn_weights = jnp.exp(
            attn_weights - jnp.max(attn_weights, axis=-1, keepdims=True)
        )
        attn_weights = attn_weights / (
            jnp.sum(attn_weights, axis=-1, keepdims=True) + 1e-8
        )

        # Weighted sum of values
        out = attn_weights @ V  # (B, H, S, head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.embed_dim)

        # Output projection
        return out @ self.W_o
