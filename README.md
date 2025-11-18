PAXJAXLIB
===========

A simple neural network implementation using JAX.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                                 INSTALLATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1.  Clone the repository:
   #   
     git clone https://github.com/paxamans/paxjaxlib.git

2.  Install in development mode with dependencies:
   #
    pip install -e .[dev]
3.
    Dependencies:
    - jax
    - jaxlib
    - numpy
    - optax

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                                 ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The library has been re-architected to strictly follow JAX functional patterns.

[ MODULE SYSTEM ]
All layers and models now inherit from `paxjaxlib.core.Module`. This base class
automatically registers the object as a JAX Pytree. This means:
-   Models are stateless data structures.
-   Parameters are stored as attributes on the object.
-   Gradients can be computed directly with respect to the model object.

[ OPTIMIZATION ]
We have integrated `optax` for optimization. The `Trainer` class now accepts
any `optax` gradient transformation.
-   State: (model, opt_state)
-   Update: optax.apply_updates(model, updates)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                                  CHANGELOG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[ 2025-11-18 ] CONFIGURATION MODERNIZATION
------------------------------------------
•   Introduced `pyproject.toml` for centralized build and tool configuration.
•   Added `ruff` for linting and formatting.
•   Added `mypy` for static type checking.
•   Configured `pytest` for testing.
•   Added GitHub Actions CI workflow (`.github/workflows/ci.yml`).
•   Added pre-commit hooks (`.pre-commit-config.yaml`).
•   Fixed `.gitignore` to correctly track `setup.py` and requirements.

[ 2025-11-18 ] ARCHITECTURE REFACTOR
------------------------------------
•   REMOVED: Custom `optimizers.py`. Replaced with `optax`.
•   REMOVED: Manual parameter management in `NeuralNetwork`.
•   ADDED: `paxjaxlib.core.Module` for automatic Pytree registration.
•   MODIFIED: `Trainer` now handles `opt_state` explicitly.
•   MODIFIED: `NeuralNetwork` and Layers now inherit from `Module`.
•   UPDATED: `examples/usage.py` to reflect the new API.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                                    USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

See `examples/usage.py` for a complete MNIST training example.

Basic Flow:
    import jax
    import optax
    from paxjaxlib import NeuralNetwork, Dense, Trainer

    # 1. Define Model
    layers = [Dense(10, 5, jax.random.PRNGKey(0))]
    model = NeuralNetwork(layers)

    # 2. Initialize Trainer
    trainer = Trainer(model, optimizer=optax.adam(1e-3))

    # 3. Train
    history = trainer.train(X, y, epochs=10)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
