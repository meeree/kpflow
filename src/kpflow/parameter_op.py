# Parameter operator from main paper. 
import torch
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
    def __init__(self, model_f, inputs, hidden, dev = 'cpu'):
        super().__init__()

        self.dev = dev
        inputs = np_to_torch(inputs).to(dev)
        hidden = np_to_torch(hidden).to(dev)
        self.shape = hidden.shape
        x_flat = inputs.reshape((-1, inputs.shape[-1]))
        z_flat = hidden.reshape((-1, hidden.shape[-1]))
        def func_param_only(params):
            return functional_call(model_f, params, (x_flat, z_flat)) # Turn model into a functor accepting parameters as an argument.

        self.model_f = func_param_only
        self.params = dict(model_f.to(dev).named_parameters())

    @torch.no_grad
    def __call__(self, q):
        # Input should be a DICTIONARY of named parameters, e.g. dict(model.named_parameters()).
        # a tuple of length one with the dictionary is also accepted.
        inp = (q,) if isinstance(q, dict) else q
        return jvp(self.model_f, (self.params,), inp)[1].reshape(self.shape)  # [B, T, H]

    @torch.no_grad
    def adjoint_call(self, q):
        q_flat = np_to_torch(q).to(self.dev).reshape((-1, q.shape[-1])) # [..., B, T, H] -> [..., H].

        # This computes vec @ J(x2).T. It contracts over all axes (e.g. time, batches, hidden) and gives something of same shape as theta.
        _, vjp_fn = vjp(self.model_f, self.params) 
        vjps = vjp_fn(q_flat)
        return vjps[0] # A dictionary.

class ParameterOperator(Operator):
    def __init__(self, model_f, inputs, hidden, dev = 'cpu'):
        super().__init__()
        self.self_adjoint = True # We are self adjoint! :)
        self.jtheta_op = JThetaOperator(model_f, inputs, hidden, dev)

    def __call__(self, q):
        return self.jtheta_op(self.jtheta_op.adjoint_call(q)) # J_theta @ J_theta^T @ q.
