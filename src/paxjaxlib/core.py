from typing import Any, Dict, List, Tuple

import jax


class Module:
    """
    Base class for all neural network modules.
    Automatically registers subclasses as JAX Pytrees.
    """

    def __init__(self):
        """Initialize the module with an empty losses list."""
        self._losses = []

    def __call__(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        jax.tree_util.register_pytree_node(
            cls,
            cls._tree_flatten,
            cls._tree_unflatten,
        )

    def add_loss(self, loss):
        """Add a regularization loss to this module."""
        self._losses.append(loss)

    def clear_losses(self):
        """Clear losses from this module and all child modules."""
        self._losses = []
        # Recursively clear losses from child modules
        for attr_name in vars(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Module):
                attr.clear_losses()
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Module):
                        item.clear_losses()

    @property
    def losses(self):
        """Return all losses from this module and its children."""
        all_losses = list(self._losses)
        # Recursively collect losses from child modules
        for attr_name in vars(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Module):
                all_losses.extend(attr.losses)
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Module):
                        all_losses.extend(item.losses)
        return all_losses

    def _tree_flatten(self) -> Tuple[List[Any], Tuple[Dict[str, Any], List[str]]]:
        """
        Flatten the module.
        """
        children = []
        children_keys = []
        aux_data = {}

        sorted_keys = sorted(vars(self).keys())

        for key in sorted_keys:
            val = getattr(self, key)

            # Determine if it should be a child (dynamic) or aux (static)
            is_child = True

            # Callables (that are not Modules) are definitely static
            if callable(val) and not isinstance(val, Module):
                is_child = False

            # Primitives are static
            elif isinstance(val, (int, float, str, bool, type(None))):
                is_child = False

            # Tuples of primitives (like shapes) are static
            elif isinstance(val, tuple) and all(
                isinstance(x, (int, float, str, bool, type(None))) for x in val
            ):
                is_child = False

            if is_child:
                children.append(val)
                children_keys.append(key)
            else:
                aux_data[key] = val

        return children, (aux_data, children_keys)

    @classmethod
    def _tree_unflatten(
        cls, aux_info: Tuple[Dict[str, Any], List[str]], children: List[Any]
    ):
        """
        Reconstruct the module.
        """
        aux_data, children_keys = aux_info
        module = object.__new__(cls)

        # Restore static data
        for key, val in aux_data.items():
            setattr(module, key, val)

        # Restore dynamic children
        if len(children) != len(children_keys):
            raise ValueError("Mismatch between children and keys during unflattening")

        for key, val in zip(children_keys, children, strict=True):
            setattr(module, key, val)

        return module
