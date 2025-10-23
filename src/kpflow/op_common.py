# Defining what an operator is in general. 
from abc import ABC, abstractmethod
import numpy as np
import math
import torch
from math import prod

np_to_torch = lambda x: x if torch.is_tensor(x) else torch.from_numpy(x)
torch_to_np = lambda x: x if not torch.is_tensor(x) else x.detach().cpu().numpy()

class Operator(ABC):
    def __init__(self, shape_in, shape_out, dev = 'cpu', self_adjoint = False, batch_first = True):
        self.shape_in = (shape_in,) if isinstance(shape_in, int) else shape_in # Enforce that shapes are always tuples for consistency!
        self.shape_out = (shape_out,) if isinstance(shape_out, int) else shape_out 
        self.dev = dev
        self.self_adjoint = self_adjoint
        self.batch_first = batch_first

    # DEFINED BY BASES CLASS #####

    @abstractmethod
    def __matvec(self, q):
        raise Exception('__matvec is not defined')

    def __rmatvec(self, q): # If not self adjoint, need to set this by hand.
        if self.self_adjoint:
            return matvec(q)
        raise Exception('__rmatvec is not defined')

    ##############################

    # shape_in -> shape_out
    def __call__(self, q):
        # For convenience, I don't enforce exact shaping (e.g. (500, 10) is same as (50, 10, 10)). 
        # This makes things like tensor products and contractions way less of a pain in the ass.
        q = q.reshape(self.shape_in)
        return __matvec(self, q).reshape(self.shape_out)

    # shape_out -> shape_in
    def adjoint_call(self, q):
        q = q.reshape(self.shape_out)
        return __rmatvec(self, q).reshape(self.shape_in)

    # [..., *shape_in] -> [..., *shape_out] if batch_first
    # [*shape_in, ...] -> [*shape_out, ...] otherwise
    def batched_call(self, q_batch):
        q_nice = q_batch.reshape((-1, *self.shape_in)) if self.batch_first else q_batch.reshape((*self.shape_in, -1)).T
        dim = 0 if self.batch_first else -1
        fn = torch.vmap(self.__matvec, in_dims = dim, out_dims = dim)
        return fn(q_nice)

    # [D, *shape_out] -> [D, *shape_in]
    def batched_adjoint_call(self, q_batch):
        q_nice = q_batch.reshape((-1, *self.shape_out)) if self.batch_first else q_batch.reshape((*self.shape_out, -1)).T
        dim = 0 if self.batch_first else -1
        fn = torch.vmap(self.__rmatvec, in_dims = dim, out_dims = dim)
        return fn(q_nice)

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

    def eigsh(self, ncomps, compute_vecs = False, tol = 1e-8):
        from scipy.sparse.linalg import eigsh
        op_sp = self.to_scipy()
        if compute_vecs:
            evals, evecs = eigsh(op_sp, k = ncomps, return_eigenvectors = True, tol = tol)
            return evals[::-1], evecs[:, ::-1].T.reshape((-1, *self.shape_in))
        return eigsh(op_sp, k = ncomps, return_eigenvectors = False, tol = tol)[::-1]

    def trace(self, nsamp = 20):
        from .trace_estimation import trace_hupp_op
        return trace_hupp_op(self, nsamp = nsamp)

    def fro_norm(self, nsamp = 20):
        return self.gram().trace(nsamp = nsamp)

    def svd(self, ncomps, trace_dims = None, compute_vecs = False, tol = 1e-8):
        # 1. Form grammian G = W W^*
        # 2. Form G_avg = partial_average(G, trace_dims) if trace_dims not None
        # 3. Use U Sigma^2 U^T = eigsh(G_avg)
        # 4. Return diag(Sigma) or U, diag(Sigma) depending on compute_vecs
        G = self.gram()
        G_avg = G if trace_dims is None else G.partial_avg(trace_dims)
        ret = G_avg.eigsh(ncomps, compute_vecs = compute_vecs, tol = tol)
        if compute_vecs:
            ret = (np.where(ret[0] < tol, 0, ret[0]), ret[1])
            return ret[0]**0.5, ret[1]
        ret = np.where(ret < tol, 0, ret)
        return ret**0.5

    def effdim(self, trace_dims, nsamp = 20, grammian = True):
        # Use some trickery: 
        # assuming P is (m, n) and we partial average n,
        # effdim_{m}(P) = m * cos_similarity(P @ P.T, Identity)
        from .trace_estimation import op_alignment
        G = self.gram() if grammian else self # For some PSD matrices X X^T, why form (X X^T)^2?
        G_avg = G.partial_trace(trace_dims)

        prod_if = lambda x: x if isinstance(x, int) else prod(x)
        m = prod_if(G_avg.shape_in) # new shape.
        Id = IdentityOperator(G_avg.shape_in)
        cos = torch_to_np(op_alignment(G_avg, Id, nsamp = nsamp))
        return m * cos**2

    def gram(self):
        G = self @ self.T() # A ComposedOperator
        G.self_adjoint = True
        return G

    def T(self):
        return TransposedOperator(self)

    def partial_avg(self, trace_dims):
        return PartialTrace(self, trace_dims, reduction = 'mean')

    def partial_trace(self, trace_dims):
        return PartialTrace(self, trace_dims, reduction = 'sum')

    def full_matrix(self):
        # Note this functions should only be used when the operator is small enough to compute!
        flat = self.flatten()
        mat = self.batched_call(torch.eye(prod(flat.shape_in)))
        return mat

    def compare(self, op2, nsamp = 20):
        return (self - op2).fro_norm(nsamp = nsamp) / max(self.fro_norm(), op2.fro_norm())

    # BASIC OPERATIONS:

    # Compose the two operators in sequence
    def __matmul__(self, other):
        return ComposedOperator(self, other)

    # Tensor/Kronecker product with another operator. This is NOT the same as op1 @ op2 above!
    def tprod(self, op2):
        return TensorProduct(self, op2, dev = self.dev)

    # Componentwise add with a scalar, tensor, or operator.
    def __add__(self, x):
        if isinstance(x, Operator):
            return Hadamard(self, x, 'add')
        return AffineTransformedOperator(self, np_to_torch(x))

    def __sub__(self, x):
        if isinstance(x, Operator):
            return Hadamard(self, x, 'sub')
        return AffineTransformedOperator(self, -np_to_torch(x))

    def __truediv__(self, x):
        if isinstance(x, Operator):
            return Hadamard(self, x, 'div')
        return AffineTransformedOperator(self, scale = 1./np_to_torch(x))

    def __mul__(self, x):
        if isinstance(x, Operator):
            return Hadamard(self, x, 'mul')
        return AffineTransformedOperator(self, scale = np_to_torch(x))
    __rmul__ = __mul__



class IdentityOperator(Operator):
    def __init__(self, shape, dev = 'cpu'):
        super().__init__(shape, shape, dev, self_adjoint = True)
    def __matvec(self, q):
        return q
    def __rmatvec(self, q):
        return q

class TensorProduct(Operator):
    def __init__(self, op1, op2, dev = 'cpu'):
        shape_in = (*op1.shape_in, *op2.shape_in)
        shape_out = (*op1.shape_out, *op2.shape_out)
        super().__init__(shape_in, shape_out, dev, self_adjoint = (op1.self_adjoint and op2.self_adjoint))
        self.op1, self.op2 = op1, op2
        self.op1_flat, self.op2_flat = self.op1.flatten(), self.op2.flatten()

    def __matvec(self, q):
        # simplest approach is to flatten and use vec(kron(A,B) vec(C)) = vec(B C A^T)
        # Key: op1_flat, op2_flat are shape (min, mout), (nin, nout)
        mat_q = q.reshape((self.op1_flat.shape_in[0], self.op2_flat.shape_in[0])) # shape (min, nin)
        q1 = self.op2_flat.batched_call(mat_q) # (min, nout)
        q2 = self.op1_flat.batched_call(q1.T).T # (mout, nout)
        return q2.reshape(self.shape_out) # Unflatten

    def __rmatvec(self, q):
        # q should be shape (op1.shape_out, op2.shape_out)
        # Key: op1_flat, op2_flat are shape (min, mout), (nin, nout)
        mat_q = q.reshape((self.op1_flat.shape_out[0], self.op2_flat.shape_out[0])) # shape (mout, nout)
        q1 = self.op2_flat.batched_adjoint_call(mat_q) # (mout, nin)
        q2 = self.op1_flat.batched_adjoint_call(q1.T).T # (min, nin)
        return q2.reshape(self.shape_in) # Unflatten

# Combine two operators together.
class Hadamard(Operator):
    def __init__(self, op1, op2, comb = 'add', dev = 'cpu'):
        assert((op1.shape_in == op2.shape_in) and (op1.shape_out == op2.shape_out))
        super().__init__(op1.shape_in, op1.shape_out, dev, self_adjoint = (op1.self_adjoint and op2.self_adjoint))
        self.op1, self.op2 = op1, op2

        self.comb = None
        if comb == 'add':
            self.comb = lambda x, y: x + y
        elif comb == 'sub':
            self.comb = lambda x, y: x - y
        elif comb == 'mul':
            self.comb = lambda x, y: x * y
        elif comb == 'div':
            self.comb = lambda x, y: x / y
        else:
            raise Exception(f'Unsupported Hadamard combination {comb}')

    def __matvec(self, q):
        return self.comb(self.op1.__matvec(q), self.op2.__matvec(q))

    def __rmatvec(self, q):
        return self.comb(self.op1.__rmatvec(q), self.op2.__rmatvec(q))

    def __str__(self):
        return f"Hadamard_{self.comb}({self.op1}, {self.op2})"

class HadamardProduct(Operator):
    def __init__(self, op1, op2, dev = 'cpu'):
        assert((op1.shape_in == op2.shape_in) and (op1.shape_out == op2.shape_out))
        super().__init__(op1.shape_in, op1.shape_out, dev, self_adjoint = (op1.self_adjoint and op2.self_adjoint))
        self.op1, self.op2 = op1, op2

    def __matvec(self, q):
        return op1(q) * op2(q)

    def adjoint_call(self, q):
        return op1.adjoint_call(q) * op2.adjoint_call(q)

class MatrixWrapper(Operator): # Just a normal matrix
    def __init__(self, W, left_mul = True, dev = 'cpu'):
        shape_in, shape_out = W.T.shape if left_mul else W.shape
        super().__init__(shape_in, shape_out, dev, self_adjoint = False)
        self.W = np_to_torch(W)
        self.mul_fn = (lambda W, x : W @ x) if left_mul else (lambda W, x : x @ W)
        self.batched_mul_fn = (lambda W, x: (W @ x.swapaxes(0,1)).swapaxes(0,1)) if left_mul else (lambda W, x: x @ W) # note batching always is in dim 0, so need to swap for batching then swap back
        
    def __matvec(self, q):
        return self.mul_fn(self.W, q.reshape(self.shape_in)).reshape(self.shape_out)

    def __rmatvec(self, q):
        return self.mul_fn(self.W.T, q.reshape(self.shape_out)).reshape(self.shape_in)

    def batched_call(self, q_batch):
        return self.batched_mul_fn(self.W, q.reshape((-1,self.shape_in))).reshape(self.shape_out)

    def batched_adjoint_call(self, q_batch):
        return self.batched_mul_fn(self.W.T, q.reshape((-1,self.shape_out))).reshape(self.shape_in)

# Takes in flattened shape_in, shape_out.
class FlatWrapper(Operator):
    def __init__(self, op):
        prod_if = lambda x: x if isinstance(x, int) else prod(x)
        shape_in_flat, shape_out_flat = prod_if(op.shape_in), prod_if(op.shape_out)
        super().__init__(shape_in_flat, shape_out_flat, op.dev, op.self_adjoint)
        self.op = op

    def __matvec(self, q):
        # [shape_in_flat] -> [self.op.shape_in] -> call -> [self.op.shape_out] -> [shape_out_flat].
        return self.op(q).reshape(self.shape_out)

    def __rmatvec(self, q):
        # [shape_out_flat] -> [self.op.shape_out] -> call -> [self.op.shape_in] -> [shape_in_flat].
        return self.op.adjoint_call(q).reshape(self.shape_in)

# Takes inputs in numpy and puts them into torch, then back into torch after calls.
class NumpyWrapper(Operator):
    def __init__(self, op):
        super().__init__(op.shape_in, op.shape_out, 'cpu', op.self_adjoint)
        self.op = op

    def __matvec(self, q):
        with torch.no_grad(): # Using numpy so why use grads.
            torch_res = self.op(torch.from_numpy(q).to(self.op.dev))
            return torch_res.cpu().numpy()

    def __rmatvec(self, q):
        with torch.no_grad(): # Using numpy so why use grads.
            torch_res = self.op.adjoint_call(torch.from_numpy(q).to(self.op.dev))
            return torch_res.cpu().numpy()

# Take partial trace or average over certain dimensions of tensor operator.
class PartialTrace(Operator):
    def __init__(self, op, trace_dims, reduction = 'sum'):
        # Create an operator that takes in inputs that are identical along one or multiple axes. 
        assert (op.shape_in == op.shape_out) # For now I'm not sure how it applies if this is not true.
        if isinstance(trace_dims, int):
            trace_dims = [trace_dims]
        avg_shape = [1 if idx in trace_dims else op.shape_in[idx] for idx in range(len(op.shape_in))] # e.g. if trace_dims = (1,), op.shape_in = (10, 20, 30), avg_shape = (10, 1, 30).
        if -1 in trace_dims: # special case.
            avg_shape[-1] = 1
        avg_shape = tuple(avg_shape)
        super().__init__(avg_shape, avg_shape, op.dev, op.self_adjoint)
        self.op = op
        self.trace_dims = trace_dims
        self.unreduced_shape = op.shape_in
        self.reduction = lambda x: x.sum(self.trace_dims) if reduction =='sum' else x.mean(self.trace_dims)

    def __matvec(self, q):
        nice_q = q.expand(self.op.shape_in) 
        unreduced_out = self.op.__matvec(nice_q)
        return self.reduction(unreduced_out).reshape(self.shape_in)

    def __rmatvec(self, q):
        nice_q = q.expand(self.op.shape_in) 
        unreduced_out = self.op.__rmatvec(nice_q)
        return self.reduction(unreduced_out).reshape(self.shape_in)

class AffineTransformedOperator(Operator):
    def __init__(self, op, scale = 1., shift = 0.):
        super().__init__(op.shape, op.shape, op.dev, op.self_adjoint)
        self.op = op
        self.scale = scale
        self.shift = shift

    def __matvec(self, q):
        return self.op(q) * self.scale + self.shift

    def __rmatvec(self, q):
        return self.op.adjoint_call(q) * self.scale + self.shift

class ComposedOperator(Operator):
    def __init__(self, op1, op2):
        super().__init__(op2.shape_in, op1.shape_out, op2.dev, False)

        # Define operator op1 * op2.
        self.op1 = op1
        self.op2 = op2

    def __matvec(self, q):
        return self.op1(self.op2(q))

    def __rmatvec(self, q):
        return self.op2.adjoint_call(self.op1.adjoint_call(q))

class TransposedOperator(Operator):
    def __init__(self, op):
        super().__init__(op.shape_out, op.shape_in, op.dev, op.self_adjoint)
        self.op = op

    def __matvec(self, q):
        return self.op.__matvec(q)

    def __rmatvec(self, q):
        return self.op.__rmatvec(q)
