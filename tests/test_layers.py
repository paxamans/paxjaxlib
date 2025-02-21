import unittest
import jax.numpy as jnp
from jax import random
from paxjaxlib.layers import Dense, Flatten, Conv2D
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

if __name__ == '__main__':
    unittest.main()


