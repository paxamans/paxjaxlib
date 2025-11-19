<p align="center">
  <h1 align="center">paxjaxlib</h1>
  <p align="center">A simple and functional neural network library built on <a href="https://github.com/google/jax">JAX</a>.</p>
</p>

<p align="center">
  <a href="https://github.com/paxamans/paxjaxlib/actions/workflows/ci.yml">
    <img src="https://github.com/paxamans/paxjaxlib/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://badge.fury.io/py/paxjaxlib">
    <img src="https://badge.fury.io/py/paxjaxlib.svg" alt="PyPI version">
  </a>
</p>

---

**paxjaxlib** is a lightweight and modular deep learning library that embraces JAX's functional programming paradigm. Just a simple, yet powerful, API for building and training neural networks.


## Installation

You can install `paxjaxlib` directly from PyPI:

```bash
pip install paxjaxlib
```

Or, for development, clone the repository and install in editable mode:

```bash
git clone https://github.com/paxamans/paxjaxlib.git
cd paxjaxlib
pip install -e .[dev]
```

## Quick Start

Here's a quick example of how to define, train, and evaluate a simple `NeuralNetwork` on dummy data.

```python
import jax
import jax.numpy as jnp
import optax
from paxjaxlib.layers import Dense
from paxjaxlib.models import NeuralNetwork
from paxjaxlib.training import Trainer

# 1. Generate dummy data
key = jax.random.PRNGKey(0)
X = jax.random.normal(key, (100, 10))
y = jax.random.normal(key, (100, 1))

# 2. Define the model
# The model is a pytree, so its parameters can be manipulated functionally.
model = NeuralNetwork([
    Dense(10, 64),
    jax.nn.relu,
    Dense(64, 1)
])

# 3. Initialize the trainer with an Optax optimizer
# The trainer manages the training loop and state.
trainer = Trainer(model, optimizer=optax.adam(1e-3))

# 4. Train the model
# The `train` method returns a history of metrics.
history = trainer.train(X, y, epochs=10, batch_size=32)
print("Training loss:", history['loss'][-1])

# 5. Make predictions
# The trained model is available at `trainer.model`.
predictions = trainer.model(X)
```

## Architecture

The core of `paxjaxlib` is the `paxjaxlib.core.Module`, which serves as the base for all layers and models. By inheriting from `Module`, classes are automatically registered as JAX pytrees.

- **Models are data**: A `NeuralNetwork` or `Dense` layer is a simple data structure. All parameters are stored as attributes.
- **Stateless updates**: Training updates produce a new, updated model object instead of modifying an existing one in place.
- **Gradients with respect to models**: You can compute gradients directly with respect to the entire model object, for example: `grads = jax.grad(loss_fn)(model, X, y)`.

## Wiki & Documentation

Coming soon

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE.
