# Defining what an operator is in general. 
from abc import ABC, abstractmethod
import numpy as np
import math
import torch
from math import prod

np_to_torch = lambda x: x if torch.is_tensor(x) else torch.from_numpy(x)
torch_to_np = lambda x: x if not torch.is_tensor(x) else x.detach().cpu().numpy()

class Operator(ABC):
    def __init__(self, shape_in, shape_out, dev = 'cpu', self_adjoint = False):
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.dev = dev
        self.self_adjoint = self_adjoint

    # shape_in -> shape_out
    @abstractmethod
    def __call__(self, q):
        raise Exception('Call is undefined')

    # shape_out -> shape_in
    def adjoint_call(self, q): # If not self adjoint, need to set this by hand.
        if self.self_adjoint:
            return self(q) # rmatvec = matvec ;)
        raise Exception('Adjoint call is undefined')

    # [D, *shape_in] -> [D, *shape_out]
    def batched_call(self, q_batch):
        return torch.vmap(self)(q_batch)

    # [D, *shape_out] -> [D, *shape_in]
    def batched_adjoint_call(self, q_batch):
        return torch.vmap(self.adjoint_call)(q_batch)

    # Get a version of the operator where shape_in, shape_out are flattened.
    def flatten(self):
        return FlatWrapper(self) 

    def to_numpy(self):
        return NumpyWrapper(self)

    def rayleigh_coef(self, q):
        Kq = self(q) # [..., B, T, H]
        return (Kq * q).sum((-3, -2, -1)) / (q * q).sum((-3, -2, -1))

    def to_scipy(self, dtype = float):
        # Convert to a scipy LinearOperator. Shape should be the shape of a typical input to __call__.
        # Note the original operator works in pytorch, allowing for cuda, while the new one will be in numpy.
        from scipy.sparse.linalg import LinearOperator
        op_np_flat = self.flatten().to_numpy()
        matmat = lambda q_vec : np.moveaxis(op_np_flat.batched_call(np.moveaxis(q_vec, -1, 0)), 0, -1) # scipy expects columns for batching, while my batching uses first dim.
        rmatmat = lambda q_vec : np.moveaxis(op_np_flat.batched_adjoint_call(np.moveaxis(q_vec, -1, 0)), 0, -1) 
        return LinearOperator(
            (op_np_flat.shape_out, op_np_flat.shape_in),
            matvec = op_np_flat, rmatvec = op_np_flat.adjoint_call, 
            matmat = matmat, rmatmat = rmatmat,
            dtype = dtype
        )

    @staticmethod
    def effrank(singular_vals, thresh):
        var = singular_vals ** 2 
        varexpl = np.cumsum(var / np.sum(var))
        if varexpl[-1] < thresh:
            raise Exception('need more components to explain variance thresh')

        i1 = np.argmax(varexpl > thresh)
        i0 = i1 - 1
        v1 = varexpl[i1]
        v0 = 0. if i1 == 0 else varexpl[i0]
        dim = 1 + (i0 + (thresh - v0) / (v1 - v0)) # Linear interpolation. Also add 1 since dimension is 1 based.
        return dim, varexpl

    def T(self):
        return TransposedOperator(self)

    def __matmul__(self, other):
        return ComposedOperator(self, other)

# Takes in flattened shape_in, shape_out.
class FlatWrapper(Operator):
    def __init__(self, op):
        shape_in_flat, shape_out_flat = prod(op.shape_in), prod(op.shape_out)
        super().__init__(shape_in_flat, shape_out_flat, op.dev, op.self_adjoint)
        self.op = op

    def __call__(self, q):
        # [shape_in_flat] -> [self.op.shape_in] -> call -> [self.op.shape_out] -> [shape_out_flat].
        non_flat = self.op(q.reshape(self.op.shape_in))
        return non_flat.reshape(self.shape_out)

    def adjoint_call(self, q):
        # [shape_out_flat] -> [self.op.shape_out] -> call -> [self.op.shape_in] -> [shape_in_flat].
        non_flat = self.op.adjoint_call(q.reshape(self.op.shape_out))
        return non_flat.reshape(self.shape_in)

# Takes inputs in numpy and puts them into torch, then back into torch after calls.
class NumpyWrapper(Operator):
    def __init__(self, op):
        super().__init__(op.shape_in, op.shape_out, op.dev, op.self_adjoint)
        self.op = op

    def __call__(self, q):
        with torch.no_grad(): # Using numpy so why use grads.
            torch_res = self.op(torch.from_numpy(q).to(self.dev))
            return torch_res.cpu().numpy()

    def adjoint_call(self, q):
        with torch.no_grad(): # Using numpy so why use grads.
            torch_res = self.op.adjoint_call(torch.from_numpy(q).to(self.dev))
            return torch_res.cpu().numpy()

    def batched_call(self, q_batch):
        return (torch.vmap(self.op)(torch.from_numpy(q_batch).to(self.dev))).cpu().numpy()

    def batched_adjoint_call(self, q_batch):
        return (torch.vmap(self.op.adjoint_call)(torch.from_numpy(q_batch).to(self.dev))).cpu().numpy()

class AveragedOperator(Operator):
    def __init__(self, op, true_shape):
        # Create an operator that takes in inputs that are identical along one or multiple axes. 
        super().__init__(op.shape_in, op.shape_out, op.dev, op.self_adjoint)
        self.op = op
        self.true_shape = true_shape # The shape we should make inputs into.

    def __call__(self, q):
        # q should have 1s in the dimensions to expand and len(q.shape) should equal len(self.true_shape).
        new_axes = [i for i in range(len(self.true_shape)) if self.true_shape[i] != q.shape[i]]
        return self.op(np_to_torch(q).expand(self.true_shape)).mean(tuple(new_axes)).reshape(q.shape)

    def adjoint_call(self, q):
        new_axes = [i for i in range(len(self.true_shape)) if self.true_shape[i] != q.shape[i]]
        return self.op.adjoint_call(np_to_torch(q).expand(self.true_shape)).mean(tuple(new_axes)).reshape(q.shape)

class ComposedOperator(Operator):
    def __init__(self, op1, op2):
        super().__init__(op2.shape_in, op1.shape_out, op2.dev, False)

        # Define operator op1 * op2.
        self.op1 = op1
        self.op2 = op2

    def __call__(self, q):
        return self.op1(self.op2(q))

    def adjoint_call(self, q):
        return self.op2.adjoint_call(self.op1.adjoint_call(q))

class TransposedOperator(Operator):
    def __init__(self, op):
        super().__init__(op.shape_out, op.shape_in, op.dev, op.self_adjoint)
        self.op = op

    def __call__(self, q):
        return self.op.adjoint_call(q)

    def adjoint_call(self, q):
        return self.op(q)
