# Parameter operator from main paper. 
import torch
from torch import nn
import numpy as np
from torch.func import vjp, jvp, functional_call

from .op_common import Operator

# Utility function
np_to_torch = lambda x: x if torch.is_tensor(x) else torch.from_numpy(x)
torch_to_np = lambda x: x if not torch.is_tensor(x) else x.detach().cpu().numpy()

# Compute the product J_theta * q for any input q over all time batches and hidden units. 
# Adjoint call performs J_theta^T * q. The composed operator J_theta @ J_theta^T is K in paper.
# Efficient tensor NTK-style implementation using python vjp and jvp functions.
class JThetaOperator(Operator):
    def __init__(self, model_f, inputs, hidden, dev = 'cpu', h_0 = None):
        nparams = sum(p.numel() for _,p in model_f.named_parameters())
        super().__init__(nparams, hidden.shape, dev)

        inputs = np_to_torch(inputs).to(dev)
        hidden = np_to_torch(hidden).to(dev)
        self.x_flat = inputs.reshape((-1, inputs.shape[-1]))

        self.h_flat = hidden.reshape((-1, hidden.shape[-1]))

        def func_param_only(params):
#            return torch.tanh(self.h_flat) @ params['weight_hh'].T + self.x_flat @ params['weight_ih'].T
            return functional_call(model_f, params, (self.x_flat, self.h_flat)) # Turn model into a functor accepting parameters as an argument.

        self.model_f = func_param_only
        self.params = {name: p.detach().clone().requires_grad_(True).to(dev) for name, p in model_f.named_parameters()}

        self.vectorize = False # Convert parameters to vectors.

    @torch.no_grad
    def _matvec(self, q):
        # The input should be a dict of named parameters, e.g. dict(model.named_parameters()).
        # Vectorized input for this call is not supported. Only for adjoint_call is this supported for now.
        inp = (q,) if isinstance(q, dict) else q
        return jvp(self.model_f, (self.params,), inp)[1].reshape(self.shape_out)  # [B, T, H]

    @torch.no_grad
    def _rmatvec(self, q):
        q_flat = np_to_torch(q).to(self.dev).reshape((-1, q.shape[-1])) # [..., B, T, H] -> [..., H].

        # This computes vec @ J(x2).T. It contracts over all axes (e.g. time, batches, hidden) and gives something of same shape as theta.
        _, vjp_fn = vjp(self.model_f, self.params) 
        vjp_out = vjp_fn(q_flat)[0] # A dictionary.
        if not self.vectorize:
            return vjp_out
        return nn.utils.parameters_to_vector(vjp_out.values()) # A vector.

class ParameterOperator(Operator):
    def __init__(self, model_f, inputs, hidden, dev = 'cpu'):
        super().__init__(hidden.shape, hidden.shape, dev, True)
        self.jtheta_op = JThetaOperator(model_f, inputs, hidden, dev)

    def _matvec(self, q):
        # Intermdiately jtheta_op produces a dict! So use _rmatvec, _matvec, not __call__, adjoint_call to not enforce shaping.
        return self.jtheta_op._matvec(self.jtheta_op._rmatvec(q)) # J_theta @ J_theta^T @ q.

    def __str__(self):
        return f"K{tuple(self.shape_in)}"
