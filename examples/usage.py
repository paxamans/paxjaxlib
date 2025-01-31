import jax.numpy as jnp
import jax.random as jrandom
from paxjaxlib.models import NeuralNetwork
from paxjaxlib.training import Trainer
from paxjaxlib.activations import relu, linear
from paxjaxlib.schedules import exponential_decay

# Set random seed
key = jrandom.PRNGKey(0)
model_key, train_key = jrandom.split(key)
# Data generation
n_samples = 64
n_features = 4  # Reduced to 4 features

# Create features
key, subkey = jrandom.split(key)
X1 = jrandom.normal(subkey, shape=(n_samples, 1))

key, subkey = jrandom.split(key)
X2 = jrandom.uniform(subkey, shape=(n_samples, 1))

X3 = jnp.sin(jnp.linspace(0, 10, n_samples)).reshape(-1, 1)

key, subkey = jrandom.split(key)
X4 = jrandom.exponential(subkey, shape=(n_samples, 1))

# Combine features
X = jnp.hstack([X1, X2, X3, X4])  # Shape will be (512, 4)

# Create target
key, subkey = jrandom.split(key)
y = (1.5 * X1 +
     2.0 * jnp.square(X2) +
     0.5 * jnp.sin(X3) +
     0.3 * X4 +
     0.1 * jrandom.normal(subkey, shape=(n_samples, 1)))  # Noise

# Ensure y is the right shape
y = y.reshape(-1, 1)  # Shape will be (512, 1)

# Create model
model = NeuralNetwork(
    layer_sizes=[n_features, 32, 1],  # Input: 4 features
    activations=[relu, linear],
    key=model_key
)

# Create trainer
learning_rate_schedule = exponential_decay(
    initial_lr=0.001,
    decay_rate=0.95,
    decay_steps=100
)

trainer = Trainer(
    model,
    optimizer="adam",
    learning_rate=0.001,
    lr_schedule=learning_rate_schedule,
    reg_lambda=0.01
)

# Training
history = trainer.train(
    X, y,
    epochs=100,
    batch_size=8,
    verbose=True
)

# Test predictions
key, subkey = jrandom.split(key)
X_test = jrandom.normal(subkey, shape=(3, n_features))  # Test data with 4 features
predictions = model.forward(X_test)
print("Predictions:", predictions)

# Save and load model
model.save("trained_model.pkl")

loaded_model = NeuralNetwork.load(
    layer_sizes=[n_features, 32, 1],
    activations=[relu, linear],
    filename="trained_model.pkl",
    key=train_key
)

loaded_predictions = loaded_model.forward(X_test)
print("Loaded Model Predictions:", loaded_predictions)
