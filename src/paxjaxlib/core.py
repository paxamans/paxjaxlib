import jax
from typing import Any, Tuple, List, Dict

class Module:
    """
    Base class for all neural network modules.
    Automatically registers subclasses as JAX Pytrees.
    """
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        jax.tree_util.register_pytree_node(
            cls,
            cls._tree_flatten,
            cls._tree_unflatten,
        )

    def _tree_flatten(self) -> Tuple[List[Any], Tuple[List[str], Dict[str, Any]]]:
        """
        Flattens the module into dynamic children (JAX arrays, Modules) and static aux_data.
        """
        children = []
        dynamic_keys = []
        static_data = {}
        
        # Sort keys to ensure deterministic order
        sorted_keys = sorted(vars(self).keys())
        
        for key in sorted_keys:
            val = getattr(self, key)
            if isinstance(val, (jax.numpy.ndarray, Module)) or \
               (isinstance(val, list) and all(isinstance(v, Module) for v in val)):
                children.append(val)
                dynamic_keys.append(key)
            else:
                static_data[key] = val
            
        return children, (dynamic_keys, static_data)

    @classmethod
    def _tree_unflatten(cls, aux_data: Tuple[List[str], Dict[str, Any]], children: List[Any]):
        """
        Reconstruct the module from dynamic children and static aux_data.
        """
        dynamic_keys, static_data = aux_data
        module = object.__new__(cls)
        
        # Restore static data
        vars(module).update(static_data)

        # Restore dynamic children
        for key, val in zip(dynamic_keys, children):
            setattr(module, key, val)
            
        return module
