import jax.numpy as jnp
from jax import jit, grad, random, value_and_grad
from typing import Callable, List, Tuple
from .models import NeuralNetwork
from .losses import mse_loss  # Import the default loss function
from .activations import relu, linear  # Import activation functions if needed
from .optimizers import SGD, Adam, AdaMax, RMSprop, Momentum, Adafactor, AdaGrad, Adadelta
from .schedules import exponential_decay, step_decay


# Move update_params outside the class as a pure function
@jit
def update_params(params, grads, learning_rate):
    return [(W - learning_rate * dW, b - learning_rate * db)
            for (W, b), (dW, db) in zip(params, grads)]

class Trainer:
    def __init__(
        self,
        model: NeuralNetwork,
        loss_fn: Callable = mse_loss,
        optimizer: str = "sgd",
        learning_rate: float = 0.01,
        lr_schedule: Callable = None,
        reg_lambda: float = 0.0
    ):        
        self.model = model
        self.reg_lambda = reg_lambda
        
        # Initialize optimizer
        if optimizer == "adam":
            self.optimizer = Adam(learning_rate)
        elif optimizer == "sgd":
            self.optimizer = SGD(learning_rate)
        elif optimizer == "adamax":
            self.optimizer = AdaMax(learning_rate)
        elif optimizer == "rmsprop":
            self.optimizer = RMSprop(learning_rate)
        elif optimizer == "momentum":
            self.optimizer = Momentum(learning_rate)
        elif optimizer == "adafactor":
            self.optimizer = Adafactor(learning_rate)
        elif optimizer == "adagrad":
            self.optimizer = AdaGrad(learning_rate)
        elif optimizer == "adadelta":
            self.optimizer = Adadelta()
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        # Learning rate schedule
        self.lr_schedule = lr_schedule
        self.current_step = 0

        # Define forward pass
        def forward(params, X):
            current_input = X
            for i, (W, b) in enumerate(params):
                current_input = jnp.dot(current_input, W) + b
                if i < len(params) - 1:  # Apply activation except for last layer
                    current_input = self.model.activations[i](current_input)
            return current_input

        # JIT compile the forward function
        self.forward = jit(forward)

        # Define loss with regularization
        def loss(params, X, y):
            y_pred = self.forward(params, X)
            return loss_fn(y_pred, y, params, self.reg_lambda)
        
        # JIT compile the loss and gradient functions
        self.loss = jit(loss)
        self.grad_fn = jit(value_and_grad(loss))

    def train_step(self, params, X, y):
        """
        Perform a single training step.

        Args:
            params: Current model parameters.
            X (jnp.ndarray): Input data.
            y (jnp.ndarray): Target labels.

        Returns:
            Tuple[float, List[Tuple]]: The loss value and updated parameters.
        """
        loss_val, grads = self.grad_fn(params, X, y)
        new_params = self.optimizer.apply_gradients(params, grads)
        
        # Update learning rate if schedule exists
        if self.lr_schedule:
            self.current_step += 1
            self.optimizer.learning_rate = self.lr_schedule(self.current_step)
        
        return loss_val, new_params

    def train(self, X: jnp.ndarray, y: jnp.ndarray, epochs: int = 100, 
             batch_size: int = 32, verbose: bool = True):
        """
        Train the model.

        Args:
            X (jnp.ndarray): Input data.
            y (jnp.ndarray): Target labels.
            epochs (int): Number of training epochs. Default is 100.
            batch_size (int): Batch size for training. Default is 32.
            verbose (bool): Whether to print training progress. Default is True.

        Returns:
            List[float]: Training history (loss values).
        """
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
