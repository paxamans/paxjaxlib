import jax.numpy as jnp
from jax import jit, grad, random, value_and_grad
from typing import Callable, List, Tuple
from .models import NeuralNetwork

def mse_loss(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> float:
    return jnp.mean((y_pred - y_true) ** 2)

# Move update_params outside the class as a pure function
@jit
def update_params(params, grads, learning_rate):
    return [(W - learning_rate * dW, b - learning_rate * db)
            for (W, b), (dW, db) in zip(params, grads)]

class Trainer:
    def __init__(self, model: NeuralNetwork, loss_fn: Callable = mse_loss, learning_rate: float = 0.01):
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn

        # Define pure functions for forward pass and loss
        def forward(params, X):
            current_input = X
            for i, (W, b) in enumerate(params):
                current_input = jnp.dot(current_input, W) + b
                if i < len(params) - 1:  # Apply activation except for last layer
                    current_input = self.model.activations[i](current_input)
            return current_input

        def loss(params, X, y):
            y_pred = forward(params, X)
            return loss_fn(y_pred, y)

        # JIT compile the functions
        self.forward = jit(forward)
        self.loss = jit(loss)
        self.grad_fn = jit(value_and_grad(loss))

    def train_step(self, params, X: jnp.ndarray, y: jnp.ndarray) -> Tuple[float, List[Tuple]]:
        loss_val, grads = self.grad_fn(params, X, y)
        # Use the pure update_params function instead of the class method
        new_params = update_params(params, grads, self.learning_rate)
        return loss_val, new_params

    def train(self, X: jnp.ndarray, y: jnp.ndarray, epochs: int = 100, 
             batch_size: int = 32, verbose: bool = True):
        n_samples = len(X)
        history = []
        params = self.model.parameters

        for epoch in range(epochs):
            key = random.PRNGKey(epoch)
            idx = random.permutation(key, n_samples)
            X_shuffled = X[idx]
            y_shuffled = y[idx]

            epoch_losses = []
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]

                loss, params = self.train_step(params, batch_X, batch_y)
                epoch_losses.append(loss)

            avg_loss = jnp.mean(jnp.array(epoch_losses))
            history.append(float(avg_loss))

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Update model parameters after training
        self.model.parameters = params
        return history
