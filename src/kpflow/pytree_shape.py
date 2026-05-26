import torch
from torch.utils import _pytree as pytree


def _safe_reshape(x, shape):
    return x.reshape(shape)


class ShapeSpec:
    def __init__(self, spec):
        # Idempotent: ShapeSpec(ShapeSpec(...)) should preserve metadata.
        if isinstance(spec, ShapeSpec):
            self.spec = spec.spec
            self.is_tensor = spec.is_tensor
            self.tree_spec = spec.tree_spec
            self.leaf_shapes = spec.leaf_shapes
            return

        # int means single tensor shape (int,)
        if isinstance(spec, int):
            spec = (spec,)

        # tuple of ints means one tensor shape
        if self._is_tensor_shape(spec):
            self.spec = tuple(spec)
            self.is_tensor = True
            self.tree_spec = None
            self.leaf_shapes = [self.spec]
            return

        # otherwise: pytree of tensor shapes
        leaves, tree_spec = pytree.tree_flatten(
            spec,
            is_leaf=self._is_tensor_shape,
        )

        self.spec = spec
        self.is_tensor = False
        self.tree_spec = tree_spec
        self.leaf_shapes = [
            tuple((leaf,) if isinstance(leaf, int) else leaf)
            for leaf in leaves
        ]

    @staticmethod
    def _is_tensor_shape(x):
        return isinstance(x, tuple) and all(isinstance(v, int) for v in x)

    @classmethod
    def from_tree(cls, x):
        """
        Infer ShapeSpec from an actual tensor pytree.

        Stores the tree structure of the tensor pytree, not the tree
        structure of the shape tuples.
        """
        if torch.is_tensor(x):
            return cls(tuple(x.shape))

        leaves, tree_spec = pytree.tree_flatten(x)

        obj = cls.__new__(cls)
        obj.spec = pytree.tree_map(lambda t: tuple(t.shape), x)
        obj.is_tensor = False
        obj.tree_spec = tree_spec
        obj.leaf_shapes = [tuple(leaf.shape) for leaf in leaves]
        return obj

    def reshape(self, x):
        if self.is_tensor:
            return _safe_reshape(x, self.spec)

        leaves, tree_spec = pytree.tree_flatten(x)
        assert tree_spec == self.tree_spec, (
            f"Tree mismatch: {tree_spec} != {self.tree_spec}"
        )

        leaves = [
            _safe_reshape(leaf, shape)
            for leaf, shape in zip(leaves, self.leaf_shapes)
        ]
        return pytree.tree_unflatten(leaves, self.tree_spec)

    def batched_reshape(self, x):
        if self.is_tensor:
            batch_shape = x.shape[: x.ndim - len(self.spec)]
            return _safe_reshape(x, (*batch_shape, *self.spec))

        leaves, tree_spec = pytree.tree_flatten(x)
        assert tree_spec == self.tree_spec, (
            f"Tree mismatch: {tree_spec} != {self.tree_spec}"
        )

        reshaped = []
        for leaf, shape in zip(leaves, self.leaf_shapes):
            batch_shape = leaf.shape[: leaf.ndim - len(shape)]
            reshaped.append(_safe_reshape(leaf, (*batch_shape, *shape)))

        return pytree.tree_unflatten(reshaped, self.tree_spec)

    def flat_batch(self, x):
        if self.is_tensor:
            batch_shape = x.shape[: x.ndim - len(self.spec)]
            return _safe_reshape(x, (-1, *self.spec)), batch_shape

        leaves, tree_spec = pytree.tree_flatten(x)
        assert tree_spec == self.tree_spec, (
            f"Tree mismatch: {tree_spec} != {self.tree_spec}"
        )

        batch_shape = leaves[0].shape[: leaves[0].ndim - len(self.leaf_shapes[0])]
        flat_leaves = [
            _safe_reshape(leaf, (-1, *shape))
            for leaf, shape in zip(leaves, self.leaf_shapes)
        ]
        return pytree.tree_unflatten(flat_leaves, self.tree_spec), batch_shape

    def unflat_batch(self, x, batch_shape):
        if self.is_tensor:
            return _safe_reshape(x, (*batch_shape, *self.spec))

        leaves, tree_spec = pytree.tree_flatten(x)
        assert tree_spec == self.tree_spec, (
            f"Tree mismatch: {tree_spec} != {self.tree_spec}"
        )

        leaves = [
            _safe_reshape(leaf, (*batch_shape, *shape))
            for leaf, shape in zip(leaves, self.leaf_shapes)
        ]
        return pytree.tree_unflatten(leaves, self.tree_spec)

    def __iter__(self):
        return iter(self.spec)

    def __len__(self):
        return len(self.spec)

    def __getitem__(self, idx):
        return self.spec[idx]

    def __eq__(self, other):
        other = ShapeSpec(other)
        return self.spec == other.spec

    def __repr__(self):
        return repr(self.spec)
