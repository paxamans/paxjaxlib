"""
Training utilities for paxjaxlib.

The :class:`Trainer` class wraps the training loop, handling batching,
shuffling, JIT-compiled gradient updates, optional validation,
early stopping, and gradient clipping.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import jax.numpy as jnp
import optax
from jax import jit, random, value_and_grad

from .losses import mse_loss
from .metrics import accuracy
from .models import NeuralNetwork


class Trainer:
    """High-level training loop for a :class:`NeuralNetwork`.

    Args:
        model: The model to train.
        loss_fn: Loss function with signature ``(y_pred, y_true) → scalar``.
            Default :func:`~paxjaxlib.losses.mse_loss`.
        optimizer: An ``optax`` gradient transformation.
            Default ``optax.adam(1e-3)``.
        key: PRNG key for shuffling and dropout.
            Default ``PRNGKey(0)``.
        metrics: A dict ``{name: fn}`` or list of callables evaluated
            after each epoch.  Default ``{"accuracy": accuracy}``.
        max_grad_norm: If set, clips gradients by their global norm to
            this value (uses ``optax.clip_by_global_norm``).

    Example::

        trainer = Trainer(
            model,
            loss_fn=categorical_crossentropy,
            optimizer=optax.adam(1e-3),
            max_grad_norm=1.0,
        )
        history = trainer.train(
            X_train, y_train,
            epochs=20,
            val_data=(X_val, y_val),
            early_stopping_patience=5,
        )
    """

    def __init__(
        self,
        model: NeuralNetwork,
        loss_fn: Callable = mse_loss,
        optimizer: Optional[optax.GradientTransformation] = None,
        key: Optional[Any] = None,
        metrics: Union[Dict[str, Callable], List[Callable], None] = None,
        max_grad_norm: Optional[float] = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.key = key if key is not None else random.PRNGKey(0)

        # Handle metrics as dict or list
        self.metrics: Union[Dict[str, Callable], List[Callable]]
        if metrics is None:
            self.metrics = {"accuracy": accuracy}
        elif isinstance(metrics, dict):
            self.metrics = metrics
        else:
            self.metrics = metrics

        # Build the optimizer chain (optionally with gradient clipping)
        base_opt = optimizer if optimizer is not None else optax.adam(1e-3)
        if max_grad_norm is not None:
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                base_opt,
            )
        else:
            self.optimizer = base_opt

        # Initialize optimizer state
        self.opt_state = self.optimizer.init(self.model)

        # JIT compile the update step
        self._update_step = jit(self._update_step_impl)

    def _loss_fn_wrapper(
        self, model: NeuralNetwork, X: jnp.ndarray, y: jnp.ndarray, key: Optional[Any]
    ):
        y_pred = model(X, key=key, training=True)
        loss = self.loss_fn(y_pred, y)
        total_loss = loss + jnp.sum(jnp.array(model.losses))
        return total_loss

    def _metrics_wrapper(self, model: NeuralNetwork, X: jnp.ndarray, y: jnp.ndarray):
        y_pred = model(X, training=False)
        result = {}
        if isinstance(self.metrics, dict):
            for metric_name, metric_fn in self.metrics.items():
                result[metric_name] = metric_fn(y_pred, y)
        else:
            for metric_fn in self.metrics:
                metric_name = (
                    metric_fn.__name__
                    if hasattr(metric_fn, "__name__")
                    else str(metric_fn)
                )
                result[metric_name] = metric_fn(y_pred, y)
        return result

    def _update_step_impl(
        self,
        model: NeuralNetwork,
        opt_state,
        X: jnp.ndarray,
        y: jnp.ndarray,
        key: Optional[Any],
    ):
        loss_val, grads = value_and_grad(self._loss_fn_wrapper, argnums=0)(
            model, X, y, key
        )
        updates, new_opt_state = self.optimizer.update(grads, opt_state, model)
        new_model = optax.apply_updates(model, updates)
        return new_model, new_opt_state, loss_val

    def _init_history(
        self, metric_names: List[str], val_data_present: bool
    ) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {"loss": []}
        for name in metric_names:
            history[name] = []
        if val_data_present:
            history["val_loss"] = []
            for name in metric_names:
                history[f"val_{name}"] = []
        return history

    def _run_epoch(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        batch_size: int,
        num_batches: int,
        key_iter: Any,
    ) -> Tuple[float, Any]:
        n_samples = X.shape[0]
        key_iter, shuffle_key = random.split(key_iter)
        permuted_indices = random.permutation(shuffle_key, n_samples)
        x_shuffled = X[permuted_indices]
        y_shuffled = y[permuted_indices]

        epoch_losses = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            batch_x = x_shuffled[start_idx:end_idx]
            batch_y = y_shuffled[start_idx:end_idx]

            key_iter, step_key = random.split(key_iter)

            self.model, self.opt_state, loss = self._update_step(
                self.model, self.opt_state, batch_x, batch_y, step_key
            )
            epoch_losses.append(loss)

        avg_epoch_loss = float(jnp.mean(jnp.array(epoch_losses)))
        return avg_epoch_loss, key_iter

    def _evaluate_validation(
        self,
        val_data: Tuple[jnp.ndarray, jnp.ndarray],
        history: Dict[str, List[float]],
    ) -> Tuple[float, Dict[str, float]]:
        X_val, y_val = val_data
        val_loss = float(self.evaluate(X_val, y_val))
        history["val_loss"].append(val_loss)
        val_metrics = self._metrics_wrapper(self.model, X_val, y_val)
        for metric_name, metric_value in val_metrics.items():
            history[f"val_{metric_name}"].append(float(metric_value))
        return val_loss, val_metrics

    def _check_early_stopping(
        self,
        val_loss: float,
        best_val_loss: float,
        patience_counter: int,
        early_stopping_patience: int,
        epoch: int,
        verbose: bool,
    ) -> Tuple[float, int, bool]:
        """Returns (new_best_val_loss, new_patience_counter, should_stop)."""
        if val_loss < best_val_loss:
            return val_loss, 0, False

        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            if verbose:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(val_loss did not improve for "
                    f"{early_stopping_patience} epochs)"
                )
            return best_val_loss, patience_counter, True
        return best_val_loss, patience_counter, False

    def train(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
        val_data: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        early_stopping_patience: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Run the training loop.

        Args:
            X: Training inputs.
            y: Training targets.
            epochs: Number of full passes over the data. Default ``100``.
            batch_size: Mini-batch size. Default ``32``.
            verbose: Print per-epoch stats. Default ``True``.
            val_data: Optional ``(X_val, y_val)`` tuple.  When provided,
                validation loss and metrics are computed at the end of
                each epoch and stored in the history under ``val_loss``,
                ``val_<metric>``.
            early_stopping_patience: If set (and ``val_data`` is
                provided), training stops when the validation loss has
                not improved for this many consecutive epochs.

        Returns:
            A dict mapping metric names to lists of per-epoch values,
            e.g. ``{"loss": [...], "accuracy": [...], "val_loss": [...]}``.
        """
        n_samples = X.shape[0]
        num_batches = (n_samples + batch_size - 1) // batch_size

        # Build metric names
        if isinstance(self.metrics, dict):
            metric_names = list(self.metrics.keys())
        else:
            metric_names = [
                m.__name__ if hasattr(m, "__name__") else str(m) for m in self.metrics
            ]

        history = self._init_history(metric_names, val_data is not None)

        # Early stopping state
        best_val_loss = float("inf")
        patience_counter = 0
        best_model = self.model
        key_iter = self.key

        for epoch in range(epochs):
            avg_epoch_loss, key_iter = self._run_epoch(
                X, y, batch_size, num_batches, key_iter
            )
            history["loss"].append(avg_epoch_loss)

            train_metrics = self._metrics_wrapper(self.model, X, y)
            for metric_name, metric_value in train_metrics.items():
                history[metric_name].append(float(metric_value))

            # Validation
            val_log = ""
            if val_data is not None:
                val_loss, val_metrics = self._evaluate_validation(val_data, history)
                val_log = f", Val Loss: {val_loss:.4f}, Val Metrics: {val_metrics}"

                # Early stopping check
                if early_stopping_patience is not None:
                    (
                        best_val_loss,
                        patience_counter,
                        should_stop,
                    ) = self._check_early_stopping(
                        val_loss,
                        best_val_loss,
                        patience_counter,
                        early_stopping_patience,
                        epoch,
                        verbose,
                    )
                    if should_stop:
                        self.model = best_model
                        break
                    if patience_counter == 0:
                        best_model = self.model

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}, "
                    f"Metrics: {train_metrics}{val_log}"
                )

        self.history = history
        return history

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """Generate predictions for the given inputs.

        Args:
            X: Input data.
        """
        return self.model(X, training=False)

    def evaluate(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Compute loss on a dataset (e.g. a validation set).

        Args:
            X: Input data.
            y: Ground-truth labels.
        """
        y_pred = self.predict(X)
        return cast(jnp.ndarray, self.loss_fn(y_pred, y))
