import unittest
import jax.numpy as jnp
from src.models import NeuralNetwork
from src.training import Trainer, mse_loss
from src.utils import relu, linear
from jax import random

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.model = NeuralNetwork(
            layer_sizes=[2, 4, 1],
            activations=[relu, linear]
        )
        self.trainer = Trainer(self.model)

    def test_mse_loss(self):
        y_pred = jnp.array([[1.0], [2.0]])
        y_true = jnp.array([[1.0], [2.0]])
        loss = mse_loss(y_pred, y_true)
        self.assertAlmostEqual(float(loss), 0.0)

    def test_train_step(self):
        X = jnp.array([[1.0, 2.0]])
        y = jnp.array([[3.0]])
        
        # Get initial parameters
        initial_params = self.model.parameters
        
        # Perform training step
        loss, new_params = self.trainer.train_step(initial_params, X, y)
        
        # Verify loss is a float
        self.assertIsInstance(float(loss), float)
        
        # Verify parameters have been updated
        for (W_before, b_before), (W_after, b_after) in zip(initial_params, new_params):
            # Check that parameters have changed
            self.assertFalse(jnp.allclose(W_before, W_after))
            self.assertFalse(jnp.allclose(b_before, b_after))

    def test_update_params(self):
        from src.training import update_params  # Import the standalone function
    
        # Create dummy parameters and gradients
        params = [(jnp.ones((2, 4)), jnp.ones(4)), (jnp.ones((4, 1)), jnp.ones(1))]
        grads = [(jnp.ones((2, 4)), jnp.ones(4)), (jnp.ones((4, 1)), jnp.ones(1))]
    
        # Update parameters
        learning_rate = 0.01
        new_params = update_params(params, grads, learning_rate)
    
        # Check that parameters were updated correctly
        for (W, b) in new_params:
            # New parameters should be less than original ones (1 - learning_rate)
            self.assertTrue(jnp.all(W < 1.0))
            self.assertTrue(jnp.all(b < 1.0))

    def test_train(self):
        # Create simple dataset
        X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y = jnp.array([[3.0], [5.0], [7.0], [9.0]])
        
        # Get initial loss
        initial_loss = float(self.trainer.loss(self.model.parameters, X, y))
        
        # Train for a few epochs
        history = self.trainer.train(
            X, y,
            epochs=5,
            batch_size=2,
            verbose=False
        )
        
        # Check that history contains the correct number of losses
        self.assertEqual(len(history), 5)
        
        # Check that final loss is less than initial loss
        final_loss = history[-1]
        self.assertLess(final_loss, initial_loss)

    def test_forward(self):
        X = jnp.array([[1.0, 2.0]])
        params = self.model.parameters
        
        # Get prediction
        y_pred = self.trainer.forward(params, X)
        
        # Check prediction shape
        self.assertEqual(y_pred.shape, (1, 1))

    def test_batch_size(self):
        # Test with different batch sizes
        X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y = jnp.array([[3.0], [5.0], [7.0], [9.0]])
        
        # Test with batch_size=1
        history1 = self.trainer.train(
            X, y,
            epochs=3,
            batch_size=1,
            verbose=False
        )
        
        # Test with batch_size=2
        history2 = self.trainer.train(
            X, y,
            epochs=3,
            batch_size=2,
            verbose=False
        )
        
        # Test with batch_size=4 (full batch)
        history3 = self.trainer.train(
            X, y,
            epochs=3,
            batch_size=4,
            verbose=False
        )
        
        # Check that all histories have the correct length
        self.assertEqual(len(history1), 3)
        self.assertEqual(len(history2), 3)
        self.assertEqual(len(history3), 3)

    def test_learning_rate(self):
        X = jnp.array([[1.0, 2.0]])
        y = jnp.array([[3.0]])
    
        # Create trainers with different learning rates
        trainer1 = Trainer(self.model, learning_rate=0.01)
        trainer2 = Trainer(self.model, learning_rate=0.1)
    
        # Get initial parameters
        params = self.model.parameters
    
        # Perform one training step with each trainer
        _, new_params1 = trainer1.train_step(params, X, y)
        _, new_params2 = trainer2.train_step(params, X, y)
    
        # The trainer with higher learning rate should make bigger parameter changes
        for i in range(len(params)):
            W1, b1 = new_params1[i]
            W2, b2 = new_params2[i]
            W_orig, b_orig = params[i]
        
            # Compare the magnitude of changes
            diff1 = jnp.abs(W1 - W_orig).mean()
            diff2 = jnp.abs(W2 - W_orig).mean()
            self.assertLess(diff1, diff2)

    def test_model_update(self):
        # Test that model parameters are actually updated after training
        X = jnp.array([[1.0, 2.0], [2.0, 3.0]])
        y = jnp.array([[3.0], [5.0]])
        
        # Store initial parameters
        initial_params = [(W.copy(), b.copy()) for W, b in self.model.parameters]
        
        # Train the model
        self.trainer.train(
            X, y,
            epochs=1,
            batch_size=2,
            verbose=False
        )
        
        # Get final parameters
        final_params = self.model.parameters
        
        # Check that parameters have been updated
        for (W_before, b_before), (W_after, b_after) in zip(initial_params, final_params):
            self.assertFalse(jnp.allclose(W_before, W_after))
            self.assertFalse(jnp.allclose(b_before, b_after))

    def test_prediction_shape(self):
        # Test that predictions maintain correct shape for different batch sizes
        params = self.model.parameters
        
        # Single sample
        X1 = jnp.array([[1.0, 2.0]])
        y_pred1 = self.trainer.forward(params, X1)
        self.assertEqual(y_pred1.shape, (1, 1))
        
        # Multiple samples
        X2 = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        y_pred2 = self.trainer.forward(params, X2)
        self.assertEqual(y_pred2.shape, (3, 1))

    def test_loss_computation(self):
        # Test loss computation with known values
        X = jnp.array([[1.0, 2.0]])
        y = jnp.array([[3.0]])
        params = self.model.parameters
        
        # Compute loss
        loss_val = self.trainer.loss(params, X, y)
        
        # Check loss properties
        self.assertIsInstance(float(loss_val), float)
        self.assertGreaterEqual(float(loss_val), 0.0)  # MSE loss should be non-negative

    def test_gradient_computation(self):
        # Test gradient computation
        X = jnp.array([[1.0, 2.0]])
        y = jnp.array([[3.0]])
        params = self.model.parameters
        
        # Compute loss and gradients
        loss_val, grads = self.trainer.grad_fn(params, X, y)
        
        # Check gradient shapes match parameter shapes
        for (W, b), (dW, db) in zip(params, grads):
            self.assertEqual(W.shape, dW.shape)
            self.assertEqual(b.shape, db.shape)

    def test_training_convergence(self):
        # Test if the model can fit a simple linear relationship
        # Generate synthetic data with clear linear relationship
        X = jnp.array([[x] for x in jnp.linspace(-1, 1, 20)])
        y = 2 * X + 1 + 0.1 * random.normal(random.PRNGKey(0), X.shape)
        
        # Create a simple model for this task
        simple_model = NeuralNetwork(
            layer_sizes=[1, 1],
            activations=[linear]
        )
        simple_trainer = Trainer(simple_model, learning_rate=0.01)
        
        # Train the model
        history = simple_trainer.train(
            X, y,
            epochs=100,
            batch_size=10,
            verbose=False
        )
        
        # Check if loss decreased significantly
        initial_loss = history[0]
        final_loss = history[-1]
        self.assertLess(final_loss, initial_loss / 2)

    def test_zero_gradient(self):
        # Test gradient behavior with perfect predictions
        X = jnp.array([[1.0, 2.0]])
        y = jnp.array([[3.0]])
        
        # Set up a scenario where prediction equals target
        params = self.model.parameters
        y_pred = self.trainer.forward(params, X)
        
        # Compute loss with perfect predictions
        loss_val = mse_loss(y_pred, y_pred)  # Using same values for prediction and target
        self.assertAlmostEqual(float(loss_val), 0.0)

    def test_batch_independence(self):
        # Test that training with different batch orders gives similar results
        X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y = jnp.array([[3.0], [5.0], [7.0], [9.0]])
        
        # Train with different batch sizes
        history1 = self.trainer.train(X, y, epochs=10, batch_size=2, verbose=False)
        history2 = self.trainer.train(X, y, epochs=10, batch_size=4, verbose=False) 

        # Final losses should be relatively close
        self.assertLess(abs(history1[-1] - history2[-1]), 1.0)  # Allow for some variation


if __name__ == '__main__':
    unittest.main()
