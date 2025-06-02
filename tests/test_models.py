import unittest
import jax.numpy as jnp
from jax import random

from paxjaxlib.models import NeuralNetwork
from paxjaxlib.activations import relu, linear
from paxjaxlib.layers import Conv2D, Flatten, Dense, Dropout 


class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        key = random.PRNGKey(0)
        key1, key2 = random.split(key)
        self.layers = [
            Dense(input_dim=2, output_dim=4, activation=relu, key=key1),
            Dense(input_dim=4, output_dim=1, activation=linear, key=key2)
        ]
        self.model = NeuralNetwork(layers=self.layers)

    def test_initialization(self):
        self.assertEqual(len(self.model.layers), 2)
        self.assertTrue(isinstance(self.model.layers[0], Dense))
        self.assertTrue(isinstance(self.model.layers[1], Dense))
        self.assertEqual(self.model.layers[0].input_dim, 2)
        self.assertEqual(self.model.layers[0].output_dim, 4)
        self.assertEqual(self.model.layers[1].input_dim, 4)
        self.assertEqual(self.model.layers[1].output_dim, 1)

    def test_forward(self):
        X = jnp.array([[1.0, 2.0]])
        output = self.model.forward(X)
        self.assertEqual(output.shape, (1, 1))

class TestConvModel(unittest.TestCase):
    def setUp(self):
        key = random.PRNGKey(0)
        key_conv, key_dense = random.split(key) # Split the key
        self.layers = [
            Conv2D(3, 16, (3, 3), relu, key_conv), # Use split key
            Flatten(),
            Dense(16*28*28, 10, linear, key_dense) # Use another split key
        ]
        self.model = NeuralNetwork(layers=self.layers)

    def test_forward(self):
        X = random.normal(random.PRNGKey(0), (1, 28, 28, 3))
        output = self.model.forward(X)
        self.assertEqual(output.shape, (1, 10))

class TestNeuralNetworkWithDropout(unittest.TestCase):
    def setUp(self):
        self.key_master = random.PRNGKey(42)
        key_init_dense1, key_init_dense2, self.key_input_data, \
            self.key_fwd_train, self.key_fwd_eval, self.key_fwd_train2 = random.split(self.key_master, 6)
        
        self.layers_with_dropout = [
            Dense(input_dim=10, output_dim=20, activation=relu, key=key_init_dense1),
            Dropout(rate=0.5), # Dropout layer with 50% rate
            Dense(input_dim=20, output_dim=5, activation=linear, key=key_init_dense2)
        ]
        self.model_with_dropout = NeuralNetwork(layers=self.layers_with_dropout)
        self.X_sample = random.normal(self.key_input_data, (3, 10)) # Batch of 3, 10 features

    def test_forward_eval_mode(self):
        """Test that in eval mode, dropout acts as an identity (for inverted dropout)."""
        output_eval = self.model_with_dropout.forward(self.X_sample, training=False, key=self.key_fwd_eval)
        self.assertEqual(output_eval.shape, (3, 5))
        
        # For a more rigorous check: output should be deterministic if keys/params are fixed
        output_eval_again = self.model_with_dropout.forward(self.X_sample, training=False, key=self.key_fwd_eval)
        self.assertTrue(jnp.array_equal(output_eval, output_eval_again),
                        "Eval mode should be deterministic for the same input and key.")

    def test_forward_train_mode(self):
        """Test that in train mode, dropout is applied and output varies with key."""
        output_train_run1 = self.model_with_dropout.forward(self.X_sample, training=True, key=self.key_fwd_train)
        self.assertEqual(output_train_run1.shape, (3, 5))

        output_train_run2 = self.model_with_dropout.forward(self.X_sample, training=True, key=self.key_fwd_train2) # Different key
        self.assertEqual(output_train_run2.shape, (3, 5))

        # With a dropout rate of 0.5, it's highly probable the outputs will differ
        self.assertFalse(jnp.allclose(output_train_run1, output_train_run2, atol=1e-6),
                         "Training with different keys should produce different outputs due to dropout.")

        # Compare train mode with eval mode - they should differ
        output_eval = self.model_with_dropout.forward(self.X_sample, training=False, key=self.key_fwd_eval)
        self.assertFalse(jnp.allclose(output_train_run1, output_eval, atol=1e-6),
                         "Training output (with dropout) should differ from evaluation output.")

    def test_forward_train_mode_no_key_raises_error(self):
        """Test that ValueError is raised if no key is provided in training with dropout."""
        with self.assertRaisesRegex(ValueError, r"NeuralNetwork\.forward requires a 'key' for Dropout layer \(index \d+, rate=[\d\.]+\) during training\."):
            self.model_with_dropout.forward(self.X_sample, training=True, key=None)

    def test_forward_train_mode_dropout_rate_zero(self):
        """Test that no key is needed if dropout rate is 0, even in training."""
        key_d1, key_d2 = random.split(random.PRNGKey(123), 2)
        layers_zero_dropout = [
            Dense(input_dim=10, output_dim=20, activation=relu, key=key_d1),
            Dropout(rate=0.0), # Dropout with zero rate
            Dense(input_dim=20, output_dim=5, activation=linear, key=key_d2)
        ]
        model_zero_dropout = NeuralNetwork(layers=layers_zero_dropout)
        try:
            # Key is None, but since dropout rate is 0, it should not be needed by Dropout layer.
            # NeuralNetwork.forward itself won't try to split a None key if the layer doesn't require it.
            model_zero_dropout.forward(self.X_sample, training=True, key=None)
        except ValueError as e:
            # This might still fail if NeuralNetwork.forward tries to split a None key *before* checking rate.
            # The current NeuralNetwork.forward implementation splits key if layer is Dropout AND training AND rate > 0.
            # So, this should pass.
            self.fail(f"Forward pass with zero-rate dropout should not raise ValueError for missing key: {e}")


if __name__ == '__main__':
    unittest.main()
