import unittest
import jax.numpy as jnp
from jax import random
from paxjaxlib.models import NeuralNetwork
from paxjaxlib.training import Trainer, mse_loss
from paxjaxlib.activations import relu, linear
from paxjaxlib.layers import Flatten, Conv2D, Dense
from paxjaxlib.optimizers import Adam

class TestTrainer(unittest.TestCase): # This would be around line 10
    def setUp(self): # <<-- THIS LINE (and all subsequent lines in the method) MUST BE INDENTED
        self.master_key = random.PRNGKey(0)
        key_init1, key_init2, self.trainer_key = random.split(self.master_key, 3)
        self.layers = [
            Dense(2, 4, relu, key_init1),
            Dense(4, 1, linear, key_init2)
        ]
        self.model = NeuralNetwork(layers=self.layers)
        self.trainer = Trainer(self.model, key=self.trainer_key) # Added key here as discussed

    def test_mse_loss(self): # <<-- THIS AND ALL OTHER METHODS ALSO INDENTED
        y_pred = jnp.array([[1.0], [2.0]])
        y_true = jnp.array([[1.0], [2.0]])
        # mse_loss is a standalone function, not part of Trainer instance.
        # If your Trainer instance uses it, that's fine.
        # If this test is for the loss function itself, call it directly:
        # from paxjaxlib.losses import mse_loss # At the top
        loss = mse_loss(y_pred, y_true)
        self.assertAlmostEqual(float(loss), 0.0)

    def test_train_step(self):
        X = jnp.array([[1.0, 2.0]])
        y = jnp.array([[3.0]])
        
        # Get initial parameters
        initial_params = self.model.parameters
        
        # Perform training step
        dummy_key_for_step = random.PRNGKey(123)
        loss, new_params = self.trainer.train_step(initial_params, X, y, dummy_key_for_step) # Pass key
        
        # Verify loss is a float
        self.assertIsInstance(float(loss), float)
        
        # Verify parameters have been updated
        for (W_before, b_before), (W_after, b_after) in zip(initial_params, new_params):
            # Check that parameters have changed
            self.assertFalse(jnp.allclose(W_before, W_after))
            self.assertFalse(jnp.allclose(b_before, b_after))

    
    def test_update_params(self):
        from paxjaxlib.training import update_params  # Corrected import
    
        # Create dummy parameters and gradients
        params = [(jnp.ones((2, 4)), jnp.ones(4)), (jnp.ones((4, 1)), jnp.ones(1))]
        grads = [(jnp.ones((2, 4)), jnp.ones(4)), (jnp.ones((4, 1)), jnp.ones(1))]
    
        # Update parameters
        learning_rate = 0.01
        new_params = update_params(params, grads, learning_rate)
    
        # Check that parameters were updated correctly
        for (W, b) in new_params:
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
        y_pred = self.trainer.predict(X, params_to_use=params, eval_key=random.PRNGKey(777)) # Use predict
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
        dummy_key_for_step = random.PRNGKey(123)
        _, new_params1 = trainer1.train_step(params, X, y, dummy_key_for_step) # Pass key
        _, new_params2 = trainer2.train_step(params, X, y, dummy_key_for_step) # Pass key (can be same for this test's purpose)
    
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
        y_pred1 = self.trainer.predict(X1, params_to_use=params, eval_key=random.PRNGKey(2))
        self.assertEqual(y_pred1.shape, (1, 1))
        
        # Multiple samples
        X2 = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        y_pred2 = self.trainer.predict(X2, params_to_use=params, eval_key=random.PRNGKey(3))
        self.assertEqual(y_pred2.shape, (3, 1))

    def test_loss_computation(self):
        # Test loss computation with known values
        X = jnp.array([[1.0, 2.0]])
        y = jnp.array([[3.0]])
        params = self.model.parameters
        
        # Compute loss
        loss_val = self.trainer.value_and_grad_fn(params, X, y, random.PRNGKey(0))[0]
        
        # Check loss properties
        self.assertIsInstance(float(loss_val), float)
        self.assertGreaterEqual(float(loss_val), 0.0)  # MSE loss should be non-negative

    def test_gradient_computation(self):
        # Test gradient computation
        X = jnp.array([[1.0, 2.0]])
        y = jnp.array([[3.0]])
        params = self.model.parameters
        
        # Compute loss and gradients
        loss_val, grads = self.trainer.value_and_grad_fn(params, X, y, random.PRNGKey(0))
        
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
        key = random.PRNGKey(0)
        key1, key2 = random.split(key)
        layers = [
            Dense(1, 1, linear, key1)
        ]
        simple_model = NeuralNetwork(layers=layers)
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
        X = jnp.array([[1.0, 2.0]])
        # Construct parameters and y such that the model's prediction is perfect
        # This might require manual setting or a simpler model/scenario
        # For simplicity, let's assume we can find such params or use a fixed output
        # For this example, let's use the output of the forward pass as the target
        params = self.model.parameters
        y_perfect_pred = self.trainer.predict(X, params_to_use=params, eval_key=random.PRNGKey(5))

        # Use value_and_grad_fn, providing a dummy key as it's required by the signature
        loss_val, grads = self.trainer.value_and_grad_fn(params, X, y_perfect_pred, random.PRNGKey(0))
    
        self.assertAlmostEqual(float(loss_val), 0.0, places=5)
        for dW, db in grads: # Assuming (dW, db) structure
            self.assertTrue(jnp.allclose(dW, jnp.zeros_like(dW), atol=1e-6))
            self.assertTrue(jnp.allclose(db, jnp.zeros_like(db), atol=1e-6))

    def test_batch_independence(self):
        # Test that training with different batch orders gives similar results
        X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y = jnp.array([[3.0], [5.0], [7.0], [9.0]])
        
        # Train with different batch sizes
        history1 = self.trainer.train(X, y, epochs=10, batch_size=2, verbose=False)
        history2 = self.trainer.train(X, y, epochs=10, batch_size=4, verbose=False) 

        # Final losses should be relatively close
        self.assertLess(abs(history1[-1] - history2[-1]), 1.0)  # Allow for some variation

    def test_adam_optimizer(self):
        key = random.PRNGKey(0)
        key1, key2 = random.split(key)
        layers = [
            Dense(2, 4, relu, key1),
            Dense(4, 1, linear, key2)
        ]
        model = NeuralNetwork(layers=layers)
        trainer = Trainer(model, optimizer="adam", learning_rate=0.001)
        
        # Perform training step and check parameter updates
        X = jnp.array([[1.0, 2.0]])
        y = jnp.array([[3.0]])
        initial_params = model.parameters # .copy() is not standard for list of tuples from JAX model params
        dummy_key_for_step = random.PRNGKey(123) # Add a dummy key
        loss, new_params = trainer.train_step(initial_params, X, y, dummy_key_for_step)
        
        # Ensure parameters have changed
        for (W_before, b_before), (W_after, b_after) in zip(initial_params, new_params):
            self.assertFalse(jnp.allclose(W_before, W_after))
            self.assertFalse(jnp.allclose(b_before, b_after))

        # Check if Adam's internal state (moments) are updated
        self.assertIsNotNone(trainer.optimizer.m)
        self.assertIsNotNone(trainer.optimizer.v)

    def test_l2_regularization(self):
        key = random.PRNGKey(0)
        key1, key2 = random.split(key)
        layers = [
            Dense(2, 4, relu, key1),
            Dense(4, 1, linear, key2)
        ]
        model = NeuralNetwork(layers=layers)
        trainer = Trainer(model, reg_lambda=0.1)
        X = jnp.array([[1.0, 2.0]])
        y = jnp.array([[3.0]])
        loss_val = self.trainer.value_and_grad_fn(model.parameters, X, y, key)[0]
        self.assertGreater(loss_val, 0)  # Loss should include L2 term

    def test_model_serialization(self):
        key = random.PRNGKey(0)
        key1, key2 = random.split(key)
        layers = [
            Dense(2, 4, relu, key1),
            Dense(4, 1, linear, key2)
        ]
        model = NeuralNetwork(layers=layers)
        model.save("test_model.pkl")
        loaded_model = NeuralNetwork.load(layers=layers, filename="test_model.pkl")
    
        self.assertEqual(len(model.parameters), len(loaded_model.parameters))
        for p_orig, p_loaded in zip(model.parameters, loaded_model.parameters):
            # Assuming parameters are (W, b) tuples
            self.assertTrue(jnp.allclose(p_orig[0], p_loaded[0])) # Compare W
            self.assertTrue(jnp.allclose(p_orig[1], p_loaded[1])) # Compare b

class TestConvTraining(unittest.TestCase):
    def setUp(self):
        key = random.PRNGKey(0)
        layers = [
            Conv2D(3, 16, (3, 3), relu, key),
            Flatten(),
            Dense(16*28*28, 10, linear, key)
        ]
        self.model = NeuralNetwork(layers=layers)
        self.trainer = Trainer(self.model, optimizer="adam")

    def test_conv_forward(self):
        X = random.normal(random.PRNGKey(0), (4, 28, 28, 3))
        params = self.model.parameters
        output = self.trainer.predict(X, params_to_use=params)
        self.assertEqual(output.shape, (4, 10))

    def test_conv_gradients(self):
        X = random.normal(random.PRNGKey(0), (4, 28, 28, 3))
        y = random.normal(random.PRNGKey(0), (4, 10))
        
        # Test gradient shapes
        _, grads = self.trainer.value_and_grad_fn(self.model.parameters, X, y, random.PRNGKey(0))
        conv_grads = grads[0]  # Gradients for Conv2D layer
        self.assertEqual(conv_grads[0].shape, (3, 3, 3, 16))  # Kernel grads
        self.assertEqual(conv_grads[1].shape, (16,))          # Bias grads

    def test_conv_training_step(self):
        X = random.normal(random.PRNGKey(0), (4, 28, 28, 3))
        y = random.normal(random.PRNGKey(0), (4, 10))
        
        initial_params = self.model.parameters
        dummy_key_for_step = random.PRNGKey(123)
        loss, new_params = self.trainer.train_step(initial_params, X, y, dummy_key_for_step) # Pass key
        
        # Verify conv parameters update
        conv_params_before = initial_params[0]
        conv_params_after = new_params[0]
        self.assertFalse(jnp.allclose(conv_params_before[0], conv_params_after[0]))
        self.assertFalse(jnp.allclose(conv_params_before[1], conv_params_after[1]))

if __name__ == '__main__':
    unittest.main()
