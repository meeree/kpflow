# Defining what an operator is in general. 
from abc import ABC, abstractmethod
import numpy as np
import math
import torch

np_to_torch = lambda x: x if torch.is_tensor(x) else torch.from_numpy(x)
torch_to_np = lambda x: x if not torch.is_tensor(x) else x.detach().cpu().numpy()

class Operator(ABC):
    def __init__(self):
        self.self_adjoint = False # By default, assume not self adjoint.

    # The input q is shape [..., B, T, H] = [..., batch count, timesteps, hidden count]. Compute action, which has same shape as q.
    @abstractmethod
    def __call__(self, q):
        pass

    def adjoint_call(self, q): # If not self adjoint, need to set this by hand.
        if self.self_adjoint:
            return self(q) # rmatvec = matvec ;)
        raise Exception('Adjoint call is undefined')

    def rayleigh_coef(self, q):
        Kq = self(q) # [..., B, T, H]
        return (Kq * q).sum((-3, -2, -1)) / (q * q).sum((-3, -2, -1))

    def to_scipy(self, shape, shape_out = None, dtype = float, can_matmat = False):
        # Convert to a scipy LinearOperator. Shape should be the shape of a typical input to __call__.
        from scipy.sparse.linalg import LinearOperator

        shape_flat = math.prod(shape)
        shape_out = shape_out if shape_out is not None else shape
        shape_out_flat = math.prod(shape_out) 
        flat_action = lambda q_flat: torch_to_np(self(q_flat.reshape(shape)).flatten())
        flat_adjoint_action = lambda q_flat: torch_to_np(self.adjoint_call(q_flat.reshape(shape_out)).flatten()) # Returns None if not defined explicitly in inherited class and not self adjoint.

        if can_matmat:
            # act on an object of shape (shape_flat, k) for some arbitrary k. Put this at start, assuming call and adjoint_call can accept something of shape (k, *shape).
            flat_matmat = lambda q_flat: torch_to_np(self(q_flat.T.reshape((-1, *shape))).reshape((-1, shape_flat))).T
            flat_rmatmat = lambda q_flat: torch_to_np(self.adjoint_call(q_flat.T.reshape((-1, *shape))).reshape((-1, shape_flat))).T

            return LinearOperator((shape_out_flat, shape_flat), flat_action, rmatvec = flat_adjoint_action, matmat = flat_matmat, rmatmat = flat_rmatmat, dtype = dtype)

        return LinearOperator((shape_out_flat, shape_flat), flat_action, rmatvec = flat_adjoint_action, dtype = dtype)

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

def check_adjoint(A, trials=5, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    m, n = A.shape
    rel_err = []
    for i in range(trials):
        x = rng.standard_normal(n) + 1j*rng.standard_normal(n) if np.iscomplexobj(A.matvec(np.ones(n))) else rng.standard_normal(n)
        y = rng.standard_normal(m) + 1j*rng.standard_normal(m) if np.iscomplexobj(A.matvec(np.ones(n))) else rng.standard_normal(m)
        lhs = np.vdot(A.matvec(x), y)      # <Ax, y>
        rhs = np.vdot(x, A.rmatvec(y))     # <x, A* y>
        rel_err.append(abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1))
    return np.stack(rel_err)

class AveragedOperator(Operator):
    def __init__(self, op, true_shape):
        # Create an operator that takes in inputs that are identical along one or multiple axes. 
        super().__init__()
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
        super().__init__()

        # Define operator op1 * op2.
        self.op1 = op1
        self.op2 = op2

    def __call__(self, q):
        return self.op1(self.op2(q))

    def adjoint_call(self, q):
        return self.op2.adjoint_call(self.op1.adjoint_call(q))

class TransposedOperator(Operator):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def __call__(self, q):
        return self.op.adjoint_call(q)

    def adjoint_call(self, q):
        return self.op(q)



class FinalTimestepOperator(Operator):
    def __init__(self, op, full_shape):
        # Create an operator that takes the final time in forward call. In adjoint call it pads with zeros.
        super().__init__()
        self.op = op
        self.full_shape = full_shape

    def __call__(self, q):
        # q is shape [..., B, T, H]. Output is shape [..., B, 1, H].
        return self.op(q)[..., :, -1:, :]

    def adjoint_call(self, q):
        # q is shape [..., B, 1, H]. Output is shape [..., B, T, H].
        # create a q_pad tensor which has zeros everywhere except at final times.
        q_pad = torch.zeros(self.full_shape)
        q_pad[..., :, -1:, :] = np_to_torch(q)
        return self.op.adjoint_call(q_pad)
