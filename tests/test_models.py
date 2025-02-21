import unittest
import jax.numpy as jnp
from jax import random
from paxjaxlib.models import NeuralNetwork
from paxjaxlib.activations import relu, linear
from paxjaxlib.layers import Conv2D, Flatten, Dense

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
        self.layers = [
            Conv2D(3, 16, (3, 3), relu, key),
            Flatten(),
            Dense(16*28*28, 10, linear, key)
        ]
        self.model = NeuralNetwork(layers=self.layers)

    def test_forward(self):
        X = random.normal(random.PRNGKey(0), (1, 28, 28, 3))
        output = self.model.forward(X)
        self.assertEqual(output.shape, (1, 10))

if __name__ == '__main__':
    unittest.main()
