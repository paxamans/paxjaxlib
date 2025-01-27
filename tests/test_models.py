import unittest
import jax.numpy as jnp
from src.models import NeuralNetwork
from src.utils import relu, linear

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.model = NeuralNetwork(
            layer_sizes=[2, 4, 1],
            activations=[relu, linear]
        )

    def test_initialization(self):
        self.assertEqual(len(self.model.layers), 2)
        self.assertEqual(self.model.layers[0].input_dim, 2)
        self.assertEqual(self.model.layers[0].output_dim, 4)
        self.assertEqual(self.model.layers[1].input_dim, 4)
        self.assertEqual(self.model.layers[1].output_dim, 1)

    def test_forward(self):
        X = jnp.array([[1.0, 2.0]])
        output = self.model.forward(X)
        self.assertEqual(output.shape, (1, 1))

if __name__ == '__main__':
    unittest.main()
