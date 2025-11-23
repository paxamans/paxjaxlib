import jax
import jax.numpy as jnp
import optax
from jax import random

from src.paxjaxlib import (
    Dense,
    NeuralNetwork,
    Trainer,
    categorical_crossentropy,
    relu,
    softmax,
)


def test_training_learnable():
    print("Generating learnable data...")
    key = random.PRNGKey(42)

    # Generate simple data: class 0 is low values, class 1 is high values
    X = jnp.concatenate(
        [random.normal(key, (100, 10)) - 2.0, random.normal(key, (100, 10)) + 2.0]
    )
    y = jnp.concatenate(
        [jnp.zeros(100, dtype=jnp.int32), jnp.ones(100, dtype=jnp.int32)]
    )

    # Shuffle
    perm = random.permutation(key, len(X))
    X = X[perm]
    y = y[perm]

    y_one_hot = jax.nn.one_hot(y, 2)

    print("Defining model...")
    model = NeuralNetwork(
        [Dense(10, 16, key, activation=relu), Dense(16, 2, key, activation=softmax)]
    )

    trainer = Trainer(
        model=model,
        loss_fn=categorical_crossentropy,
        optimizer=optax.adam(0.01),
        key=key,
        metrics={
            "acc": lambda y_p, y_t: jnp.mean(jnp.argmax(y_p, -1) == jnp.argmax(y_t, -1))
        },
    )

    print("Training...")
    history = trainer.train(X, y_one_hot, epochs=20, verbose=True)

    final_acc = history["acc"][-1]
    print(f"Final Accuracy: {final_acc}")

    if final_acc > 0.9:
        print("SUCCESS: Model learned the task!")
    else:
        print("FAILURE: Model failed to learn.")


if __name__ == "__main__":
    test_training_learnable()
