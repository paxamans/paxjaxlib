from typing import Sequence, Callable, List, Optional, Tuple
import jax.numpy as jnp
from jax import random, tree_util as tree # tree_util might not be used here after corrections
import pickle
import inspect # For inspecting layer forward method signatures

class NeuralNetwork:
    def __init__(self, layers: List):
        self.layers = layers

    def forward(self, X: jnp.ndarray, training: bool = False, key: Optional[random.PRNGKey] = None) -> jnp.ndarray:
        """
        Forward pass through the network.
        Args:
            X (jnp.ndarray): Input data.
            training (bool): Whether the network is in training mode (e.g., for Dropout).
            key (Optional[random.PRNGKey]): Base PRNGKey. If layers require keys (like Dropout),
                                             this key will be split and passed.
        Returns:
            jnp.ndarray: Output of the network.
        """
        current_input = X
        iter_key = key # Key for iterating through layers

        for layer_idx, layer in enumerate(self.layers): # Added enumerate for more robust key splitting if needed
            sig = inspect.signature(layer.forward)
            call_args = {}

            if 'training' in sig.parameters:
                call_args['training'] = training

            if 'key' in sig.parameters:
                # Check if the layer is a Dropout layer (or any other layer that needs a key during training)
                # Using globals().get('Dropout') to avoid circular import if Dropout is in .layers
                # A direct import `from .layers import Dropout` and `isinstance(layer, Dropout)` is cleaner.
                is_dropout_layer = isinstance(layer, globals().get('Dropout', object)) # Fallback to object if Dropout not found

                if is_dropout_layer and training and hasattr(layer, 'rate') and layer.rate > 0.0:
                    if iter_key is None:
                        raise ValueError(
                            f"NeuralNetwork.forward requires a 'key' for Dropout layer (index {layer_idx}, rate={layer.rate}) "
                            "during training."
                        )
                    iter_key, subkey = random.split(iter_key)
                    call_args['key'] = subkey
                # Add conditions for other keyed layers if necessary

            current_input = layer.forward(current_input, **call_args)
        return current_input

    def save(self, filename: str):
        params = self.parameters
        with open(filename, 'wb') as f:
            pickle.dump(params, f)

    @classmethod
    def load(cls, layers: List, filename: str): # Corrected signature (no 'key')
        with open(filename, 'rb') as f:
            params = pickle.load(f)
        model = cls(layers)
        model.parameters = params
        return model

    @property
    def parameters(self):
        return [layer.parameters for layer in self.layers if hasattr(layer, 'parameters') and layer.parameters]

    @parameters.setter
    def parameters(self, params_list: List[Tuple]):
        param_idx = 0
        for layer in self.layers:
            if hasattr(layer, 'parameters') and layer.parameters:
                if param_idx < len(params_list):
                    layer.parameters = params_list[param_idx]
                    param_idx += 1
                else:
                    raise ValueError("Mismatch between number of layer parameters and provided params_list.")