import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from src.models import NeuralNetwork
from src.training import Trainer
from src.utils import relu, linear

def run_sine_regression():
    # Generate sample data
    X = jnp.linspace(-5, 5, 1000).reshape(-1, 1)
    y = jnp.sin(X) + 0.1 * random.normal(random.PRNGKey(0), X.shape)

    # Create model
    model = NeuralNetwork(
        layer_sizes=[1, 5600, 5600, 1],
        activations=[relu, relu, linear]
    )

    # Create trainer and train
    trainer = Trainer(model, learning_rate=0.01)
    history = trainer.train(X, y, epochs=100, batch_size=32)

    # Plot results
    plot_results(X, y, model, history)

def plot_results(X, y, model, history):
    y_pred = model.forward(X)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X, y, label='True', alpha=0.5)
    plt.plot(X, y_pred, 'r-', label='Predicted')
    plt.title('Predictions')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_sine_regression()
