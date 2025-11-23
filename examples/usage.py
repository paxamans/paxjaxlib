import jax
import jax.numpy as jnp
import optax
import tensorflow as tf  # For TFDS
import tensorflow_datasets as tfds
from jax import random

# --- Library Imports ---
from paxjaxlib import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    NeuralNetwork,
    Trainer,
    categorical_crossentropy,
    relu,
    softmax,
)

# --- Configuration ---
tf.config.experimental.set_visible_devices([], "GPU")


# --- 1. Load and Preprocess MNIST Data ---
def load_and_preprocess_data():
    print("Loading MNIST dataset...")
    ds_builder = tfds.builder("mnist")
    ds_builder.download_and_prepare()
    datasets = ds_builder.as_dataset(split=["train", "test"], as_supervised=True)
    train_ds, test_ds = datasets[0], datasets[1]

    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    train_ds = (
        train_ds.map(preprocess)
        .cache()
        .shuffle(10000)
        .batch(128)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = test_ds.map(preprocess).batch(1024).prefetch(tf.data.AUTOTUNE)

    # Convert to JAX arrays
    X_train_list, y_train_list = [], []
    for image, label in tfds.as_numpy(train_ds):
        X_train_list.append(image)
        y_train_list.append(label)
    X_train_jax = jnp.concatenate(X_train_list)
    y_train_jax = jnp.concatenate(y_train_list)

    X_test_list, y_test_list = [], []
    for image, label in tfds.as_numpy(test_ds):
        X_test_list.append(image)
        y_test_list.append(label)
    X_test_jax = jnp.concatenate(X_test_list)
    y_test_jax = jnp.concatenate(y_test_list)

    # One-hot encode labels
    num_classes = 10
    y_train_jax_one_hot = jax.nn.one_hot(y_train_jax, num_classes)
    y_test_jax_one_hot = jax.nn.one_hot(y_test_jax, num_classes)

    return X_train_jax, y_train_jax_one_hot, X_test_jax, y_test_jax_one_hot, y_test_jax


X_train, y_train, X_test, y_test_one_hot, y_test_labels = load_and_preprocess_data()

# --- 2. Define Model Architecture ---
key_master = random.PRNGKey(42)
key_layers_init, key_trainer_init = random.split(key_master)

# Split keys for layer initializations
keys_for_layers = random.split(key_layers_init, 4)


def create_model(keys):
    return NeuralNetwork(
        [
            Conv2D(
                input_channels=1,
                output_channels=16,
                kernel_size=(3, 3),
                activation=relu,
                key=keys[0],
                padding="SAME",
            ),
            Conv2D(
                input_channels=16,
                output_channels=32,
                kernel_size=(3, 3),
                activation=relu,
                key=keys[1],
                padding="SAME",
            ),
            Flatten(),
            Dense(
                input_dim=28 * 28 * 32,
                output_dim=128,
                activation=relu,
                key=keys[2],
            ),
            Dropout(rate=0.5),
            Dense(input_dim=128, output_dim=10, activation=softmax, key=keys[3]),
        ]
    )


model = create_model(keys_for_layers)
print("\nModel architecture defined.")

# --- 3. Initialize Trainer ---
trainer = Trainer(
    model=model,
    loss_fn=categorical_crossentropy,
    optimizer=optax.adam(learning_rate=0.001),
    key=key_trainer_init,
)
print("\nTrainer initialized with Adam optimizer.")

# --- 4. Train the Model ---
print("\nStarting training...")
epochs = 2  # Quick test
batch_size = 128
history = trainer.train(
    X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=True
)

print("\nTraining finished.")
print("Training history (loss per epoch):", [f"{loss:.4f}" for loss in history["loss"]])

# --- 5. Evaluate the Model ---
print("\nEvaluating model...")
y_pred_test_probs = trainer.predict(X_test)
predicted_classes = jnp.argmax(y_pred_test_probs, axis=1)

accuracy = jnp.mean(predicted_classes == y_test_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# --- 6. Save and Load Model ---
model_filename = "mnist_cnn_model_paxjaxlib.pkl"
print(f"\nSaving model to {model_filename}...")
trainer.model.save(model_filename)

print(f"Loading model from {model_filename}...")
# Create a new model instance with the same architecture
# We can use new random keys since weights will be overwritten
loaded_model = create_model(random.split(random.PRNGKey(0), 4))
loaded_model.load(model_filename)
print("Model loaded.")

loaded_y_pred = loaded_model(X_test, training=False)
loaded_predicted_classes = jnp.argmax(loaded_y_pred, axis=1)
loaded_accuracy = jnp.mean(loaded_predicted_classes == y_test_labels)
print(f"Loaded Model Test Accuracy: {loaded_accuracy * 100:.2f}%")

if jnp.allclose(accuracy, loaded_accuracy):
    print("Original and loaded model accuracies match. Save/Load successful!")
else:
    print("Accuracies MISMATCH.")
