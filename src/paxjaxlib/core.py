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

    def _tree_flatten(self) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Flatten the module into leaves (JAX arrays/Modules) and aux_data (static config).
        By default, we treat all instance attributes as leaves if they are JAX types or Modules,
        and everything else as static aux_data.
        However, for simplicity and robustness in this custom implementation,
        we will treat ALL attributes as potential leaves if they are not callables or dunder methods.
        JAX's tree_flatten will recursively handle them.
        
        Actually, a safer approach for a custom library without complex filtering 
        is to assume everything in __dict__ is a child node unless specified otherwise.
        But to distinguish static config (int, str, tuple of ints) from parameters (arrays),
        we rely on JAX's default behavior for standard types.
        
        The tricky part is distinguishing what should be static (aux_data) vs dynamic (children).
        In Equinox, this is done with dataclasses and type filtering.
        Here, we'll use a simple heuristic:
        - JAX arrays, Modules, and lists/dicts/tuples of them are children.
        - Primitives (int, float, str, bool, None) and tuples of them are aux_data.
        
        Wait, JAX's register_pytree_node expects:
        flatten(x) -> (children, aux_data)
        unflatten(aux_data, children) -> x
        
        If we put everything in children, JAX handles it. But static args to __init__ need to be aux_data
        if we want to reconstruct the object.
        BUT, we are not reconstructing via __init__ in unflatten usually, we use __new__ and set state.
        
        Let's use the standard pattern:
        Children = vars(self).values()
        Aux = vars(self).keys()
        """
        children = []
        aux_data = []
        keys = []
        
        # Sort keys to ensure deterministic order
        sorted_keys = sorted(vars(self).keys())
        
        for key in sorted_keys:
            val = getattr(self, key)
            children.append(val)
            keys.append(key)
            
        return children, keys

    @classmethod
    def _tree_unflatten(cls, keys: List[str], children: List[Any]):
        """
        Reconstruct the module.
        """
        module = object.__new__(cls)
        for key, val in zip(keys, children):
            setattr(module, key, val)
        return module
