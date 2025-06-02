
import jax
import jax.numpy as jnp
from jax import random
import tensorflow_datasets as tfds
import tensorflow as tf # For TFDS, and to prevent it from grabbing GPU

# --- Library Imports ---
# Assuming your library is in a directory called 'paxjaxlib'
# and you are running this script from the directory containing 'paxjaxlib'
from paxjaxlib import ( # <--- UPDATED IMPORT
    NeuralNetwork,
    Dense,
    Conv2D,
    Flatten,
    Dropout,
    relu,
    softmax,
    categorical_crossentropy,
    Adam, # Using Adam optimizer
    Trainer
)

# --- Configuration ---
# Prevent TensorFlow from allocating GPU memory, allowing JAX to use it.
tf.config.experimental.set_visible_devices([], 'GPU')

# --- 1. Load and Preprocess MNIST Data ---
def load_and_preprocess_data():
    print("Loading MNIST dataset...")
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    datasets = ds_builder.as_dataset(split=['train', 'test'], as_supervised=True)
    train_ds, test_ds = datasets[0], datasets[1]

    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
        # Ensure image is NHWC (Batch, Height, Width, Channels)
        # MNIST images are (H, W, C=1) by default from tfds
        return image, label

    train_ds = train_ds.map(preprocess).cache().shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess).batch(1024).prefetch(tf.data.AUTOTUNE)

    # Convert to JAX arrays
    # For simplicity, load all into memory. For larger datasets, iterate.
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
    
    print(f"X_train shape: {X_train_jax.shape}, y_train_one_hot shape: {y_train_jax_one_hot.shape}")
    print(f"X_test shape: {X_test_jax.shape}, y_test_one_hot shape: {y_test_jax_one_hot.shape}")
    
    return X_train_jax, y_train_jax_one_hot, X_test_jax, y_test_jax_one_hot, y_test_jax # Return original y_test for easy acc check

X_train, y_train, X_test, y_test_one_hot, y_test_labels = load_and_preprocess_data()

# --- 2. Define Model Architecture ---
key_master = random.PRNGKey(42)
key_layers_init, key_trainer_init = random.split(key_master)

# Split keys for layer initializations
keys_for_layers = random.split(key_layers_init, 4) # For 2 Conv2D and 2 Dense layers

# Input shape for MNIST: (28, 28, 1)
# After Conv1 (3x3, SAME, 16 filters): (28, 28, 16)
# After Conv2 (3x3, SAME, 32 filters): (28, 28, 32)
# After Flatten: 28 * 28 * 32 = 25088
# This assumes no pooling. If pooling is added, this calculation changes.

layers = [
    Conv2D(input_channels=1, output_channels=16, kernel_size=(3,3), activation=relu, key=keys_for_layers[0], padding="SAME"),
    Conv2D(input_channels=16, output_channels=32, kernel_size=(3,3), activation=relu, key=keys_for_layers[1], padding="SAME"),
    Flatten(),
    Dense(input_dim=28*28*32, output_dim=128, activation=relu, key=keys_for_layers[2]),
    Dropout(rate=0.5), # Dropout rate of 50%
    Dense(input_dim=128, output_dim=10, activation=softmax, key=keys_for_layers[3]) # 10 classes for MNIST
]

model = NeuralNetwork(layers)
print("\nModel architecture defined.")
for i, layer_obj in enumerate(model.layers): # Renamed 'layer' to 'layer_obj' to avoid conflict
    print(f"Layer {i}: {layer_obj.__class__.__name__}", end="")
    if hasattr(layer_obj, 'parameters') and layer_obj.parameters:
        if isinstance(layer_obj.parameters, tuple) and len(layer_obj.parameters) > 0 and hasattr(layer_obj.parameters[0], 'shape'):
             print(f", W shape: {layer_obj.parameters[0].shape}", end="")
        if isinstance(layer_obj.parameters, tuple) and len(layer_obj.parameters) > 1 and hasattr(layer_obj.parameters[1], 'shape'):
             print(f", b shape: {layer_obj.parameters[1].shape}", end="")
    if hasattr(layer_obj, 'rate'):
        print(f", rate: {layer_obj.rate}", end="")
    print()


# --- 3. Initialize Trainer ---
trainer = Trainer(
    model=model,
    loss_fn=categorical_crossentropy,
    optimizer="adam", # Use Adam optimizer
    learning_rate=0.001,
    key=key_trainer_init # Pass a key to the trainer for its own PRNG needs (shuffling, dropout keys)
)
print("\nTrainer initialized with Adam optimizer.")

# --- 4. Train the Model ---
print("\nStarting training...")
epochs = 5 # Keep epochs low for a quick demo
batch_size = 128 
history = trainer.train(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=True)

print("\nTraining finished.")
print("Training history (loss per epoch):", [f"{l:.4f}" for l in history])

# --- 5. Evaluate the Model ---
print("\nEvaluating model...")
y_pred_test_probs = trainer.predict(X_test)
predicted_classes = jnp.argmax(y_pred_test_probs, axis=1)

accuracy = jnp.mean(predicted_classes == y_test_labels) 
print(f"Test Accuracy: {accuracy * 100:.2f}%")

test_loss = trainer.loss(trainer.model.parameters, X_test, y_test_one_hot)
print(f"Test Loss: {test_loss:.4f}")


# --- 6. Show Some Predictions (Optional) ---
print("\nSample predictions (first 5 test images):")
for i in range(5):
    true_label = y_test_labels[i]
    pred_label = predicted_classes[i]
    print(f"Image {i+1}: True Label = {true_label}, Predicted Label = {pred_label} {'(Correct)' if true_label == pred_label else '(Incorrect)'}")

# --- 7. Save and Load Model (Demonstration) ---
model_filename = "mnist_cnn_model_paxjaxlib.pkl" # Changed filename slightly
print(f"\nSaving model parameters to {model_filename}...")
trainer.model.save(model_filename)

print(f"Loading model parameters from {model_filename}...")
key_layers_load, _ = random.split(random.PRNGKey(99)) 
keys_for_load_layers = random.split(key_layers_load, 4)

loaded_layers_list = [ # Renamed to avoid conflict
    Conv2D(input_channels=1, output_channels=16, kernel_size=(3,3), activation=relu, key=keys_for_load_layers[0], padding="SAME"),
    Conv2D(input_channels=16, output_channels=32, kernel_size=(3,3), activation=relu, key=keys_for_load_layers[1], padding="SAME"),
    Flatten(),
    Dense(input_dim=28*28*32, output_dim=128, activation=relu, key=keys_for_load_layers[2]),
    Dropout(rate=0.5),
    Dense(input_dim=128, output_dim=10, activation=softmax, key=keys_for_load_layers[3])
]

loaded_model = NeuralNetwork.load(layers=loaded_layers_list, filename=model_filename)
print("Model loaded.")

loaded_y_pred_test_probs = loaded_model.forward(X_test, training=False) 
loaded_predicted_classes = jnp.argmax(loaded_y_pred_test_probs, axis=1)
loaded_accuracy = jnp.mean(loaded_predicted_classes == y_test_labels)
print(f"Loaded Model Test Accuracy: {loaded_accuracy * 100:.2f}%")

if jnp.allclose(accuracy, loaded_accuracy):
    print("Original and loaded model accuracies match. Save/Load successful!")
else:
    print("Accuracies MISMATCH. Check save/load logic.")

print("\n--- MNIST Example Finished ---")
