import jax.numpy as jnp
from jax import jit, grad, random, value_and_grad
from typing import Callable, List, Tuple, Optional 
import inspect 

from .models import NeuralNetwork
from .losses import mse_loss 
from .optimizers import SGD, Adam, AdaMax, RMSprop, Momentum, Adafactor, AdaGrad, Adadelta
from .layers import Dropout

@jit
def update_params(params, grads, learning_rate): # This is SGD specific, optimizers handle their own logic
    # This helper might be redundant if optimizers are used directly.
    # For SGD, it's fine. Other optimizers have their own state.
    return [(W - learning_rate * dW, b - learning_rate * db)
            for (W, b), (dW, db) in zip(params, grads)]

class Trainer:

    def __init__(
        self,
        model: NeuralNetwork,
        loss_fn: Callable = mse_loss,
        optimizer: str = "sgd", # Consider passing an optimizer instance directly
        learning_rate: float = 0.01, # Optimizer instances will take LR
        lr_schedule: Callable = None,
        reg_lambda: float = 0.0,
        key: Optional[random.PRNGKey] = None # Main PRNGKey for the Trainer
    ):
        self.model = model
        self.loss_fn_orig = loss_fn # Store original loss_fn
        self.reg_lambda = reg_lambda
        
        # Initialize PRNGKey for the trainer
        self.key = key if key is not None else random.PRNGKey(0) # Default key

        # Initialize optimizer
        opt_classes = {
            "sgd": SGD, "adam": Adam, "adamax": AdaMax, "rmsprop": RMSprop,
            "momentum": Momentum, "adafactor": Adafactor, "adagrad": AdaGrad,
            "adadelta": Adadelta
        }
        if optimizer.lower() in opt_classes:
            if optimizer.lower() == "adadelta":
                self.optimizer = opt_classes[optimizer.lower()]() # Adadelta might not take LR initially
            else:
                self.optimizer = opt_classes[optimizer.lower()](learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        self.lr_schedule = lr_schedule
        self.current_step = 0 # Global step counter for LR schedule and key generation

        # This internal forward function is JIT-compiled and handles explicit parameters
        # and the training/dropout_key logic.
        def _model_forward_with_explicit_params( # Definition
            params_list: List[Tuple],
            X_batch: jnp.ndarray,
            training_mode: bool,
            dropout_key: Optional[random.PRNGKey] # << TO THIS (matches call)
        ):
            current_input = X_batch
            param_idx = 0
            # Make sure internal logic uses dropout_key now if it was using dropout_iter_key
            current_dropout_key_for_loop = dropout_key # Assign to the loop variable

            for i_layer, layer_obj in enumerate(self.model.layers):
                layer_sig = inspect.signature(layer_obj.forward)
                call_kwargs = {}

                # Pass explicit W, b if layer has parameters
                if hasattr(layer_obj, 'parameters') and layer_obj.parameters:
                    if param_idx >= len(params_list):
                        raise ValueError("Parameter list length mismatch during forward pass.")
                    
                    # Assuming parameters are (W,b) for layers that have them
                    # This needs to be robust if layers have different param structures
                    # For Dense/Conv2D, this is (W, b)
                    if 'W' in layer_sig.parameters and 'b' in layer_sig.parameters:
                         call_kwargs['W'], call_kwargs['b'] = params_list[param_idx]
                    else: # Layer has parameters, but not W,b named args (e.g. BatchNorm scale/shift)
                          # This part would need extension if such layers are added
                          # For now, assume layers with params take W,b in forward (when explicit)
                          # If not, their forward should just use self.W, self.b,
                          # but here we are passing params_list explicitly.
                          # This part of the design is tricky: how layers accept explicit params.
                          # The current Dense/Conv2D layers take optional W,b.
                        pass # Assume for now W,b are the only params passed this way

                    param_idx += 1

                if 'training' in layer_sig.parameters:
                    call_kwargs['training'] = training_mode
            
                if 'key' in layer_sig.parameters:
                    is_dropout = isinstance(layer_obj, Dropout)
                    if is_dropout and training_mode and hasattr(layer_obj, 'rate') and layer_obj.rate > 0.0:
                        if current_dropout_key_for_loop is None: # Check the loop key
                            raise ValueError("Trainer's forward pass requires a dropout_key for Dropout layers in training.")
                        current_dropout_key_for_loop, subkey_for_dropout = random.split(current_dropout_key_for_loop) # Split loop key
                        call_kwargs['key'] = subkey_for_dropout
            
                current_input = layer_obj.forward(current_input, **call_kwargs)
            return current_input
    
        self.forward_internal = jit(_model_forward_with_explicit_params, static_argnums=(2,))

        def loss_for_grad(
            params_for_loss: List[Tuple], 
            X_for_loss: jnp.ndarray, 
            y_for_loss: jnp.ndarray, 
            dropout_key_for_loss: random.PRNGKey # This name is fine (dropout_key_for_loss)
        ):
            y_pred = self.forward_internal( # Call the internal forward pass
                params_for_loss, X_for_loss, training_mode=True, dropout_key=dropout_key_for_loss # Pass it as dropout_key
            )
            l2_params = params_for_loss if self.reg_lambda > 0 else None
            return self.loss_fn_orig(y_pred, y_for_loss, l2_params, self.reg_lambda)
    
        self.value_and_grad_fn = jit(value_and_grad(loss_for_grad))

    def loss(self, params: List[Tuple], X: jnp.ndarray, y: jnp.ndarray, dropout_key: Optional[random.PRNGKey] = None) -> float:
        """
        Computes the loss for the given parameters and data, typically in evaluation mode.
        """
        # Use forward_internal with training_mode=False
        y_pred = self.forward_internal(params, X, training_mode=False, dropout_key=dropout_key)
        
        # Use the original loss function, possibly with regularization
        l2_params = params if self.reg_lambda > 0 else None
        return self.loss_fn_orig(y_pred, y, l2_params, self.reg_lambda)

    def train_step(self, current_params: List[Tuple], X_batch: jnp.ndarray, y_batch: jnp.ndarray, step_dropout_key: random.PRNGKey):
        loss_val, grads = self.value_and_grad_fn(current_params, X_batch, y_batch, step_dropout_key)
        updated_params = self.optimizer.apply_gradients(current_params, grads)
        
        # Update learning rate if schedule exists
        # self.current_step is incremented in the train loop *after* this call for the current step
        if self.lr_schedule:
            # Adam and other optimizers might have their own 't' (step count).
            # If lr_schedule is for this optimizer, it should use optimizer's 't' or a shared step.
            # For simplicity, Trainer's current_step is used.
            effective_lr = self.lr_schedule(self.current_step) # Use step *before* increment for current schedule
            if hasattr(self.optimizer, 'learning_rate'): # Not all optimizers expose LR this way (e.g. Adadelta)
                 self.optimizer.learning_rate = effective_lr
            elif hasattr(self.optimizer, 'lr'): # Some might use 'lr'
                 self.optimizer.lr = effective_lr


        return loss_val, updated_params

    def train(self, X: jnp.ndarray, y: jnp.ndarray, epochs: int = 100, 
             batch_size: int = 32, verbose: bool = True):
        n_samples = X.shape[0]
        num_batches = (n_samples + batch_size - 1) // batch_size # Handles partial last batch
        
        history = []
        current_model_params = self.model.parameters

        # Split the main trainer key for per-epoch shuffling and per-step dropout keys
        epoch_shuffling_key_base = random.fold_in(self.key, 1)
        step_dropout_key_base = random.fold_in(self.key, 2)

        for epoch in range(epochs):
            # Shuffle data for each epoch
            epoch_shuffle_key = random.fold_in(epoch_shuffling_key_base, epoch)
            permuted_indices = random.permutation(epoch_shuffle_key, n_samples)
            X_shuffled = X[permuted_indices]
            y_shuffled = y[permuted_indices]

            epoch_losses = []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_samples)
                batch_X_data = X_shuffled[start_idx:end_idx]
                batch_y_data = y_shuffled[start_idx:end_idx]

                # Generate a unique dropout key for this training step
                # self.current_step is the global batch/optimizer step counter
                current_dropout_key = random.fold_in(step_dropout_key_base, self.current_step)
                
                loss, current_model_params = self.train_step(
                    current_model_params, batch_X_data, batch_y_data, current_dropout_key
                )
                epoch_losses.append(loss)
                self.current_step += 1 # Increment global step counter

            self.model.parameters = current_model_params # Update model with new params after each epoch

            avg_epoch_loss = jnp.mean(jnp.array(epoch_losses))
            history.append(float(avg_epoch_loss))

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
        
        return history

    def predict(self, X: jnp.ndarray, params_to_use: Optional[List[Tuple]] = None, eval_key: Optional[random.PRNGKey] = None) -> jnp.ndarray:
        """
        Make predictions using the model in evaluation mode (e.g., dropout disabled).

        Args:
            X (jnp.ndarray): Input data.
            params_to_use (Optional): Model parameters. If None, uses current self.model.parameters.
            eval_key (Optional): PRNGKey if any layer in eval mode might need it (uncommon for Dropout).

        Returns:
            jnp.ndarray: Model predictions.
        """
        if params_to_use is None:
            params_to_use = self.model.parameters
        
        # Call the internal forward pass with training_mode=False.
        # Dropout, as implemented, does not use the key when training_mode=False.
        return self.forward_internal(params_to_use, X, training_mode=False, dropout_key=eval_key)
