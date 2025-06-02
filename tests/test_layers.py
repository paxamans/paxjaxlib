import unittest
import jax.numpy as jnp
from jax import random

from paxjaxlib.layers import Dense, Flatten, Conv2D, Dropout
from paxjaxlib.activations import relu

class TestDense(unittest.TestCase):  # Updated class name
    def setUp(self):
        key = random.PRNGKey(0)
        self.layer = Dense(2, 3, activation=lambda x: x, key=key)  # Updated class name

    def test_initialization(self):
        self.assertEqual(self.layer.input_dim, 2)
        self.assertEqual(self.layer.output_dim, 3)
        self.assertEqual(self.layer.W.shape, (2, 3))
        self.assertEqual(self.layer.b.shape, (3,))

    def test_forward(self):
        X = jnp.array([[1.0, 2.0]])
        output = self.layer.forward(X)
        self.assertEqual(output.shape, (1, 3))

class TestConv2D(unittest.TestCase):
    def setUp(self):
        key = random.PRNGKey(0)
        self.conv_layer = Conv2D(
            input_channels=3,
            output_channels=4,
            kernel_size=(3, 3),
            activation=relu,
            key=key,
            padding="SAME"
        )

    def test_initialization(self):
        self.assertEqual(self.conv_layer.W.shape, (3, 3, 3, 4))
        self.assertEqual(self.conv_layer.b.shape, (4,))

    def test_forward(self):
        X = random.normal(random.PRNGKey(0), (1, 28, 28, 3))  # NHWC format
        output = self.conv_layer.forward(X)
        self.assertEqual(output.shape, (1, 28, 28, 4))

class TestFlatten(unittest.TestCase):
    def test_forward(self):
        flatten = Flatten()
        X = random.normal(random.PRNGKey(0), (2, 14, 14, 16))
        output = flatten.forward(X)
        self.assertEqual(output.shape, (2, 14*14*16))
        
class TestDropout(unittest.TestCase):
    def test_initialization(self):
        layer = Dropout(rate=0.5)
        self.assertEqual(layer.rate, 0.5)
        with self.assertRaises(ValueError):
            Dropout(rate=1.0) # Rate must be < 1.0
        with self.assertRaises(ValueError):
            Dropout(rate=-0.1)

    def test_forward_eval_mode(self):
        layer = Dropout(rate=0.5)
        X = jnp.ones((5, 10))
        key = random.PRNGKey(0) # Key for consistency, though not used by Dropout in eval
        output = layer.forward(X, training=False, key=key)
        self.assertTrue(jnp.array_equal(output, X), "Output should be same as input in eval mode.")

    def test_forward_train_mode_rate_zero(self):
        layer_zero_rate = Dropout(rate=0.0)
        X = jnp.ones((5, 10))
        key = random.PRNGKey(0)
        output = layer_zero_rate.forward(X, training=True, key=key) # Key not strictly needed if rate is 0
        self.assertTrue(jnp.array_equal(output, X), "Output should be same if rate is 0, even in training.")

    def test_forward_train_mode_with_dropout(self):
        rate = 0.5
        layer = Dropout(rate=rate)
        X = jnp.ones((100, 200)) # Use a larger array to see statistical properties
        key = random.PRNGKey(0)
        
        output = layer.forward(X, training=True, key=key)
        
        self.assertEqual(output.shape, X.shape)
        
        # Check some values are zeroed out
        self.assertTrue(jnp.any(output == 0.0), "Some elements should be zero after dropout.")
        
        # Check non-zero values are scaled by 1 / keep_prob
        keep_prob = 1.0 - rate
        expected_scale_factor = 1.0 / keep_prob
        
        non_zero_mask = (output != 0.0)
        num_non_zero_actual = jnp.sum(non_zero_mask)
        num_total_elements = X.size
        
        # Expected number of non-zero elements is roughly num_total * keep_prob
        # Use a reasonable delta for statistical variation
        self.assertAlmostEqual(num_non_zero_actual / num_total_elements, keep_prob, delta=0.05,
                               msg="Proportion of non-zero elements is off.")

        # For X=jnp.ones, non-zero output elements should be exactly expected_scale_factor
        if num_non_zero_actual > 0:
            non_zero_elements = output[non_zero_mask]
            self.assertTrue(jnp.allclose(non_zero_elements, expected_scale_factor),
                            "Non-zero elements are not scaled correctly.")

    def test_forward_train_mode_different_keys_produce_different_masks(self):
        layer = Dropout(rate=0.5)
        X = jnp.ones((5, 10))
        key1 = random.PRNGKey(0)
        key2 = random.PRNGKey(1) # A different key
        
        output1 = layer.forward(X, training=True, key=key1)
        output2 = layer.forward(X, training=True, key=key2)
        
        # Outputs should be different due to different dropout masks from different keys
        self.assertFalse(jnp.array_equal(output1, output2), 
                         "Different keys should produce different dropout masks.")

    def test_forward_train_mode_no_key_raises_error(self):
        layer = Dropout(rate=0.5) # Rate > 0
        X = jnp.ones((2, 2))
        with self.assertRaisesRegex(ValueError, "Dropout layer requires a PRNGKey"):
            layer.forward(X, training=True, key=None)

        # Should NOT raise if rate is 0.0, even if key is None
        layer_zero_rate = Dropout(rate=0.0)
        try:
            layer_zero_rate.forward(X, training=True, key=None)
        except ValueError:
            self.fail("Dropout with rate 0.0 should not require a key in training mode, even if key is None.")

if __name__ == '__main__':
    unittest.main()


