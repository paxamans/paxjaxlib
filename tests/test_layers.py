import unittest
import jax.numpy as jnp
from src.layers import Layer

class TestLayer(unittest.TestCase):
    def setUp(self):
        self.layer = Layer(2, 3)

    def test_initialization(self):
        self.assertEqual(self.layer.input_dim, 2)
        self.assertEqual(self.layer.output_dim, 3)
        self.assertEqual(self.layer.W.shape, (2, 3))
        self.assertEqual(self.layer.b.shape, (3,))

    def test_forward(self):
        X = jnp.array([[1.0, 2.0]])
        output = self.layer.forward(X)
        self.assertEqual(output.shape, (1, 3))

if __name__ == '__main__':
    unittest.main()
