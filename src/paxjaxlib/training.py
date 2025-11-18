from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from jax import jit, random, value_and_grad

from .losses import mse_loss
from .models import NeuralNetwork


class Trainer:
    def __init__(
        self,
        model: NeuralNetwork,
        loss_fn: Callable = mse_loss,
        optimizer: optax.GradientTransformation = optax.adam(1e-3),
        key: Optional[random.PRNGKey] = None
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.key = key if key is not None else random.PRNGKey(0)
        
        # Initialize optimizer state
        self.opt_state = self.optimizer.init(self.model)

        # JIT compile the update step
        self._update_step = jit(self._update_step_impl)

    def _loss_fn_wrapper(self, model: NeuralNetwork, X: jnp.ndarray, y: jnp.ndarray, key: random.PRNGKey):
        y_pred = model(X, key=key, training=True)
        return self.loss_fn(y_pred, y)

    def _update_step_impl(self, model: NeuralNetwork, opt_state, X: jnp.ndarray, y: jnp.ndarray, key: random.PRNGKey):
        loss_val, grads = value_and_grad(self._loss_fn_wrapper)(model, X, y, key)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, model)
        new_model = optax.apply_updates(model, updates)
        return new_model, new_opt_state, loss_val

    def train(self, X: jnp.ndarray, y: jnp.ndarray, epochs: int = 100, 
             batch_size: int = 32, verbose: bool = True):
        n_samples = X.shape[0]
        num_batches = (n_samples + batch_size - 1) // batch_size
        
        history = []
        
        # Split keys
        key_iter = self.key

        for epoch in range(epochs):
            key_iter, shuffle_key = random.split(key_iter)
            permuted_indices = random.permutation(shuffle_key, n_samples)
            X_shuffled = X[permuted_indices]
            y_shuffled = y[permuted_indices]

            epoch_losses = []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_samples)
                batch_X = X_shuffled[start_idx:end_idx]
                batch_y = y_shuffled[start_idx:end_idx]

                key_iter, step_key = random.split(key_iter)
                
                self.model, self.opt_state, loss = self._update_step(
                    self.model, self.opt_state, batch_X, batch_y, step_key
                )
                epoch_losses.append(loss)

            avg_epoch_loss = jnp.mean(jnp.array(epoch_losses))
            history.append(float(avg_epoch_loss))

            if verbose and (epoch + 1) % 1 == 0: # Print every epoch for better feedback in short runs
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
        
        return history

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """Predict using the current model state."""
        return self.model(X, training=False)
    
    def evaluate(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """Compute loss on validation set."""
        y_pred = self.predict(X)
        return self.loss_fn(y_pred, y)
