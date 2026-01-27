# Defining what an operator is in general. 
from abc import ABC, abstractmethod
import numpy as np
import math
import torch
from math import prod

np_to_torch = lambda x: torch.tensor(x) #x if torch.is_tensor(x) else torch.from_numpy(x)
torch_to_np = lambda x: x if not torch.is_tensor(x) else x.detach().cpu().numpy()

class Operator(ABC):
    _vmap = True

    def __init__(self, shape_in, shape_out, dev = 'cpu', self_adjoint = False, batch_first = True, force_shape = True):
        # Enforce that shapes are always tuples for consistency!
        self.shape_in = tuple((shape_in,) if isinstance(shape_in, int) else shape_in)
        self.shape_out = tuple((shape_out,) if isinstance(shape_out, int) else shape_out)
        self.dev = dev
        self.self_adjoint = self_adjoint
        self.batch_first = batch_first
        self.shapes = (self.shape_out, self.shape_in) # Out then in, to agree with intution of matrices (m,n) -> input size n, output size m
        self.force_shape = force_shape

    # DEFINED BY BASES CLASS #####

    @abstractmethod
    def _matvec(self, q):
        raise Exception('_matvec is not defined')

    def _rmatvec(self, q): # If not self adjoint, need to set this by hand.
        if self.self_adjoint:
            return self._matvec(q)
        raise Exception('_rmatvec is not defined')

    ##############################

    def _debug(self, *args):
        pass

    # shape_in -> shape_out
    @torch.no_grad
    def __call__(self, q):
        # For convenience, I don't enforce exact shaping (e.g. (500, 10) is same as (50, 10, 10)). 
        # This makes things like tensor products and contractions way less of a pain in the ass.
        self._debug('matvec, input shape = ', q.shape, 'operator shape = ', self.shape_in, '->', self.shape_out) # nop unless debug set (so no slowdown at all)
        if not self.force_shape:
            return self._matvec(q)
        q = q.reshape(self.shape_in)
        return self._matvec(q).reshape(self.shape_out)

    # shape_out -> shape_in
    @torch.no_grad
    def adjoint_call(self, q):
        self._debug('rmatvec, input shape = ', q.shape, 'operator.T shape = ', self.shape_out, '->', self.shape_in) # nop unless debug set (so no slowdown at all)
        if not self.force_shape:
            return self._rmatvec(q)
        q = q.reshape(self.shape_out)
        return self._rmatvec(q).reshape(self.shape_in)

    # [..., *shape_in] -> [..., *shape_out] if batch_first
    # [*shape_in, ...] -> [*shape_out, ...] otherwise
    def batched_call(self, q_batch):
        self._debug('batched_call, input shape = ', q_batch.shape, 'operator shape = ', self.shape_in, '->', self.shape_out, '; vmap = ', self._vmap) # nop unless debug set (so no slowdown at all)
        q_nice = q_batch.reshape((-1, *self.shape_in)) if self.batch_first else q_batch.reshape((*self.shape_in, -1)).T
        if self._vmap:
            dim = 0 if self.batch_first else -1
            fn = torch.vmap(self._matvec, in_dims = dim, out_dims = dim)
            return fn(q_nice)

        res = []
        itr = q_nice if self.batch_first else q_nice.moveaxis(-1, 0)
        for qi in itr:
            res.append(self._matvec(qi))
        return torch.stack(res, 0 if self.batch_first else -1)


    # [D, *shape_out] -> [D, *shape_in]
    def batched_adjoint_call(self, q_batch):
        self._debug('batched_adjoint_call, input shape = ', q_batch.shape, 'operator.T shape = ', self.shape_out, '->', self.shape_in, '; vmap = ', self._vmap) # nop unless debug set (so no slowdown at all)
        q_nice = q_batch.reshape((-1, *self.shape_out)) if self.batch_first else q_batch.reshape((*self.shape_out, -1)).T
        if self._vmap:
            dim = 0 if self.batch_first else -1
            fn = torch.vmap(self._rmatvec, in_dims = dim, out_dims = dim)
            return fn(q_nice)

        res = []
        itr = q_nice if self.batch_first else q_nice.moveaxis(-1, 0)
        for qi in itr:
            res.append(self._rmatvec(qi))
        return torch.stack(res, 0 if self.batch_first else -1)

    @property
    def T(self):
        return TransposedOperator(self)

    # Get a version of the operator where shape_in, shape_out are flattened.
    def flatten(self):
        shape_in_flat, shape_out_flat = prod(self.shape_in), prod(self.shape_out)
        return self.reshape(shape_in_flat, shape_out_flat)

    def to_numpy(self):
        return NumpyWrapper(self)

    def rayleigh_coef(self, q):
        Kq = self(q) # [..., B, T, H]
        return ((Kq * q).sum((-3, -2, -1)) / (q * q).sum((-3, -2, -1))).item()

    def to_scipy(self, dtype = float):
        # Convert to a scipy LinearOperator. Shape should be the shape of a typical input to __call__.
        # Note the original operator works in pytorch, allowing for cuda, while the new one will be in numpy.
        from scipy.sparse.linalg import LinearOperator
        op_np_flat = self.flatten().to_numpy()
        matmat = lambda q_vec : np.moveaxis(op_np_flat.batched_call(np.moveaxis(q_vec, -1, 0)), 0, -1) # scipy expects columns for batching, while my batching uses first dim.
        rmatmat = lambda q_vec : np.moveaxis(op_np_flat.batched_adjoint_call(np.moveaxis(q_vec, -1, 0)), 0, -1) 
        return LinearOperator(
            (op_np_flat.shape_out[0], op_np_flat.shape_in[0]),
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

    def trace(self, nsamp = 21):
        from .trace_estimation import trace_hupp_op
        return trace_hupp_op(self, nsamp = nsamp)

    def fro_norm(self, nsamp = 21):
        squared = self.gram().trace(nsamp = nsamp)
        squared = max(0., squared) # Clamp if negative. If we're getting small negative values (e.g. -1e-6) this indicates the norm is zero, it's just random variance. 
        return squared**0.5

    def alignment(self, op, nsamp = 21):
        from .trace_estimation import op_alignment
        return op_alignment(self, op, nsamp = nsamp)

    def svd(self, ncomps, keep_dims = None, compute_vecs = False, tol = 1e-8):
        # 1. Form grammian G = W W^*
        # 2. Form G_avg = partial_average(G, keep_dims) if keep_dims not None
        # 3. Use U Sigma^2 U^T = eigsh(G_avg)
        # 4. Return diag(Sigma) or U, diag(Sigma) depending on compute_vecs
        G = self.gram()
        G_avg = G
        if keep_dims is not None:
            keep_dims = (keep_dims,) if isinstance(keep_dims, int) else keep_dims
            trace_dims = tuple([dim for dim in range(len(self.shape_in)) if not (dim in keep_dims or (dim == len(self.shape_in)-1 and -1 in keep_dims))])
            G_avg = G.partial_avg(trace_dims)

        ret = G_avg.eigsh(ncomps, compute_vecs = compute_vecs, tol = tol)
        if compute_vecs:
            ret = (np.where(ret[0] < tol, 0, ret[0]), ret[1])
            return ret[0]**0.5, ret[1]
        ret = np.where(ret < tol, 0, ret)
        return ret**0.5

    def op_norm(self, keep_dims = None, tol = 1e-8):
        return self.svd(1, keep_dims = keep_dims, compute_vecs = False, tol = tol)[0]

    def effdim(self, keep_dims=None, nsamp = 21, ratio = False, grammian = True):
        # Use some trickery: 
        # assuming P is (m, n) and we partial average n,
        # effdim_{m}(P) = m * cos_similarity(P @ P.T, Identity)
        from .trace_estimation import op_alignment
        if keep_dims is None:
            keep_dims = (i for i in range(len(self.shape_in)))
        keep_dims = (keep_dims,) if isinstance(keep_dims, int) else keep_dims
        trace_dims = tuple([dim for dim in range(len(self.shape_in)) if not (dim in keep_dims or (dim == len(self.shape_in)-1 and -1 in keep_dims))])

        G = self.gram() if grammian else self # For some PSD matrices X X^T, why form (X X^T)^2?
        G_avg = G.partial_trace(trace_dims)

        m = prod(G_avg.shape_in)
        Id = IdentityOperator(G_avg.shape_in)
        cos = torch_to_np(op_alignment(G_avg, Id, nsamp = nsamp))**2
        return cos if ratio else m * cos

    def effdims(self, keep_dims_list = None, **effdim_kwargs):
        # Return partial effective dimension of operator over all possible dims.
        # OR: Pass a list of keep_dims and return effdim for all those, i.e. run it multiple times for convenience.
        dims = []
        keep_dims_iter = range(len(self.shape_in)) if keep_dims_list is None else keep_dims_list 
        for keep_dim in keep_dims_iter:
            dims.append(self.effdim(keep_dim, **effdim_kwargs))
        return tuple(dims)

    def gram(self):
        return GramOperator(self) # self @ self.T

    def symm_part(self):
        return SymmetricPartOperator(self) # (self + self.T) / 2

    def reshape(self, new_shape_in, new_shape_out=None):
        if new_shape_out is None:
            return OperatorView(self, new_shape_in, new_shape_in) 
        return OperatorView(self, new_shape_in, new_shape_out) 

    def like(self, op): # Reshape to be same shape as a new operator
        return OperatorView(self, op.shape_in, op.shape_out)

    def tprod_like(self, op, op_match_shape): # A convenient way to make tensor products and make shapes match :)
        return self.tprod(op).like(op_match_shape)

    def partial_avg(self, trace_dims):
        if trace_dims == (): # Do nothing
            return self
        return PartialTrace(self, trace_dims, reduction = 'mean')

    def partial_trace(self, trace_dims):
        if trace_dims == (): # Do nothing
            return self
        return PartialTrace(self, trace_dims, reduction = 'sum')

    def full_matrix(self):
        # Note this functions should only be used when the operator is small enough to compute!
        flat = self.flatten()
        mat = self.batched_call(torch.eye(flat.shape_in[0]))
        return mat.squeeze().T

    def compare(self, op2, nsamp = 21, method = 'rel', atol = 1e-8):
        nm = (self - op2).fro_norm(nsamp = nsamp)
        if method == 'abs':
            return nm

        # Roll back to atol if both operators are close to zero.
        return nm / max(atol, self.fro_norm(nsamp = nsamp), op2.fro_norm(nsamp = nsamp))

    def set_debug(self, val = True):
        if val:
            setattr(Operator, '_debug', lambda self, *args: print("DEBUG :", self.__class__.__name__, *args))
        else:
            setattr(Operator, '_debug', lambda self, *args: None)

    @classmethod
    def toggle_vmap(cls, val = None):
        val = val if val is not None else (not cls._vmap)
        cls._vmap = val

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
    def _matvec(self, q):
        return q
    def _rmatvec(self, q):
        return q

    def __str__(self):
        return f"I_{self.shape_in}"

class TensorProduct(Operator):
    def __init__(self, op1, op2, dev = 'cpu'):
        shape_in = (*op1.shape_in, *op2.shape_in)
        shape_out = (*op1.shape_out, *op2.shape_out)
        super().__init__(shape_in, shape_out, dev, self_adjoint = (op1.self_adjoint and op2.self_adjoint))
        self.op1, self.op2 = op1, op2
        self.op1_flat, self.op2_flat = self.op1.flatten(), self.op2.flatten()

    def _matvec(self, q):
        # simplest approach is to flatten and use vec(kron(A,B) vec(C)) = vec(B C A^T)
        # Key: op1_flat, op2_flat are shape (min, mout), (nin, nout)
        mat_q = q.reshape((self.op1_flat.shape_in[0], self.op2_flat.shape_in[0])) # shape (min, nin)
        q1 = self.op2_flat.batched_call(mat_q) # (min, nout)
        q2 = self.op1_flat.batched_call(q1.T).T # (mout, nout)
        return q2.reshape(self.shape_out) # Unflatten

    def _rmatvec(self, q):
        # q should be shape (op1.shape_out, op2.shape_out)
        # Key: op1_flat, op2_flat are shape (min, mout), (nin, nout)
        mat_q = q.reshape((self.op1_flat.shape_out[0], self.op2_flat.shape_out[0])) # shape (mout, nout)
        q1 = self.op2_flat.batched_adjoint_call(mat_q) # (mout, nin)
        q2 = self.op1_flat.batched_adjoint_call(q1.T).T # (min, nin)
        return q2.reshape(self.shape_in) # Unflatten

    def __str__(self):
        return f"({self.op1} \u2297 {self.op2})"

# Combine two operators together.
class Hadamard(Operator):
    def __init__(self, op1, op2, comb = 'add', dev = 'cpu'):
        assert ((op1.shape_in == op2.shape_in) and (op1.shape_out == op2.shape_out)), f"(shape_out, shape_in) not the same: op1 {op1.shapes} != op2 {op2.shapes}"
        super().__init__(op1.shape_in, op1.shape_out, dev, self_adjoint = (op1.self_adjoint and op2.self_adjoint))
        self.op1, self.op2 = op1, op2

        self.comb_str = {'add': '+', 'sub': '-', 'mul': '\u2297', 'div': '\u2298'}[comb]
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

    def _matvec(self, q):
        return self.comb(self.op1(q), self.op2(q))

    def _rmatvec(self, q):
        return self.comb(self.op1.adjoint_call(q), self.op2.adjoint_call(q))

    def __str__(self):
        return f"({self.op1} {self.comb_str} {self.op2})"

class MatrixWrapper(Operator): # Just a normal matrix
    def __init__(self, W, left_mul = True, dev = 'cpu'):
        shape_in, shape_out = W.T.shape if left_mul else W.shape
        super().__init__(shape_in, shape_out, dev, self_adjoint = False)
        self.W = np_to_torch(W)
        self.mul_fn = (lambda W, x : W @ x) if left_mul else (lambda W, x : x @ W)
        self.batched_mul_fn = (lambda W, x: (W @ x.swapaxes(0,1)).swapaxes(0,1)) if left_mul else (lambda W, x: x @ W) # note batching always is in dim 0, so need to swap for batching then swap back
        
    def _matvec(self, q):
        return self.mul_fn(self.W, q.reshape(self.shape_in)).reshape(self.shape_out)

    def _rmatvec(self, q):
        return self.mul_fn(self.W.T, q.reshape(self.shape_out)).reshape(self.shape_in)

    def batched_call(self, q_batch):
        return self.batched_mul_fn(self.W, q.reshape((-1,self.shape_in))).reshape(self.shape_out)

    def batched_adjoint_call(self, q_batch):
        return self.batched_mul_fn(self.W.T, q.reshape((-1,self.shape_out))).reshape(self.shape_in)

    def __str__(self):
        return f"mat({tuple(self.W.shape)})" 

# Takes in flattened shape_in, shape_out.
class OperatorView(Operator):
    def __init__(self, op, new_shape_in, new_shape_out):
#        assert (prod(new_shape_in) == prod(op.shape_in) and prod(new_shape_out) == prod(op.shape_out)), f"Can't create a view from shapes {op.shapes} to (new_shape_in, new_shape_out)"
        super().__init__(new_shape_in, new_shape_out, op.dev, op.self_adjoint)
        self.op = op

    def _matvec(self, q):
        # [new_shape_in] -> [self.op.shape_in] -> call -> [self.op.shape_out] -> [new_shape_out].
        return self.op(q).reshape(self.shape_out)

    def _rmatvec(self, q):
        # [new_shape_in] -> [self.op.shape_out] -> call -> [self.op.shape_in] -> [new_shape_out].
        return self.op.adjoint_call(q).reshape(self.shape_in)

    def __str__(self):
        return f"{self.op}.view({tuple(self.shape_in)}, {tuple(self.shape_out)})"

# Takes inputs in numpy and puts them into torch, then back into torch after calls.
class NumpyWrapper(Operator):
    def __init__(self, op):
        super().__init__(op.shape_in, op.shape_out, 'cpu', op.self_adjoint)
        self.op = op

    def _matvec(self, q):
        with torch.no_grad(): # Using numpy so why use grads.
            torch_res = self.op(torch.from_numpy(q).to(self.op.dev))
            return torch_res.cpu().numpy()

    def _rmatvec(self, q):
        with torch.no_grad(): # Using numpy so why use grads.
            torch_res = self.op.adjoint_call(torch.from_numpy(q).to(self.op.dev))
            return torch_res.cpu().numpy()

    def __str__(self):
        return f"{self.op}.np()"

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
        self.reduction_type = reduction
        self.reduction = lambda x: x.sum(self.trace_dims) if reduction =='sum' else x.mean(self.trace_dims)

    def _matvec(self, q):
        nice_q = q.expand(self.op.shape_in) 
        unreduced_out = self.op(nice_q)
        return self.reduction(unreduced_out).reshape(self.shape_in)

    def _rmatvec(self, q):
        nice_q = q.expand(self.op.shape_in) 
        unreduced_out = self.op.adjoint_call(nice_q)
        return self.reduction(unreduced_out).reshape(self.shape_in)

    def __str__(self):
        if self.reduction_type == 'mean':
            return f"mean_{trace_dims}({self.op})"
        return f"tr_{trace_dims}({self.op})"

class AffineTransformedOperator(Operator):
    def __init__(self, op, scale = 1., shift = 0.):
        super().__init__(op.shape_in, op.shape_out, op.dev, op.self_adjoint)
        self.op = op
        self.scale = scale
        self.shift = shift

    def _matvec(self, q):
        return self.op(q) * self.scale + self.shift

    def _rmatvec(self, q):
        return self.op.adjoint_call(q) * self.scale + self.shift

    def __str__(self):
        if scale == 1.:
            return f"{self.op} + {self.shift}"
        if shift == 0.:
            return f"{self.scale} * {self.op}"
        return f"({self.scale} * {self.op} + {self.shift})"

class ComposedOperator(Operator):
    def __init__(self, op1, op2):
        assert (op1.shape_in == op2.shape_out), f"Shapes do not compose: op1.shape_in {op1.shape_in} != op2.shape_out {op2.shape_out}"
        super().__init__(op2.shape_in, op1.shape_out, op2.dev, False)

        # Define operator op1 * op2.
        self.op1 = op1
        self.op2 = op2

    def _matvec(self, q):
        return self.op1(self.op2(q))

    def _rmatvec(self, q):
        return self.op2.adjoint_call(self.op1.adjoint_call(q))

    def batched_call(self, q):
        return self.op1.batched_call(self.op2.batched_call(q))

    def batched_adjoint_call(self, q):
        return self.op2.batched_adjoint_call(self.op1.batched_adjoint_call(q))

    def __str__(self):
        return f"({self.op1} \u2218 {self.op2})"

class GramOperator(Operator): # op @ op.T. Made this it's own class just to have pretty printing.
    def __init__(self, op):
        super().__init__(op.shape_out, op.shape_out, dev = op.dev, self_adjoint = True)
        self.op = op

    def _matvec(self, q):
        return self.op(self.op.adjoint_call(q))

    def batched_call(self, q):
        return self.op.batched_call(self.op.batched_adjoint_call(q))

    def __str__(self):
        return f"{self.op}.grammian()"

    def effdim(self, keep_dims, nsamp = 21):
        return super().effdim(keep_dims, nsamp, grammian = False) # So effdim(GramOperator(op)) = effdim(op). This is stylistic mostly. 

class SymmetricPartOperator(Operator): # op + op.T
    def __init__(self, op):
        assert(op.shape_in == op.shape_out)
        super().__init__(op.shape_in, op.shape_in, dev = op.dev, self_adjoint = True)
        self.op = op

    def _matvec(self, q):
        return (self.op(q) + self.op.adjoint_call(q)) * 0.5

    def __str__(self):
        return f"SymPart({self.op})"

class TransposedOperator(Operator):
    def __init__(self, op):
        super().__init__(op.shape_out, op.shape_in, op.dev, op.self_adjoint, force_shape = op.force_shape)
        self.op = op

    def _matvec(self, q):
        return self.op.adjoint_call(q)

    def _rmatvec(self, q):
        return self.op(q)

    @property
    def T(self):
        return self.op # Replaced TransposedOperator(TransposedOperator(op)) with op

    def __str__(self):
        return f"{self.op}.T"

# Given x, y tensors, defines the map q |-> <q, x>_F * y
# This basically generalizes the outer product xy^T for vectors (1-tensors) x and y.
# In bra-ket notation it is |y><x|
class Projector(Operator):
    def __init__(self, x, y = None):
        self_adjoint = (x == y) 
        if y is None:
            self_adjoint = True
            y = x # x x^T grammian default if no y provided.

        super().__init__(x.shape, y.shape, dev = x.device, self_adjoint = self_adjoint)
        self.x, self.y = x, y

    def _matvec(self, q):
        inr = (self.x * q).sum()
        return inr * self.y

    def _rmatvec(self, q):
        inr = (self.y * q).sum()
        return inr * self.x

    def __str__(self):
        if self.self_adjoint:
            return "x x^T"
        return "x y^T"
