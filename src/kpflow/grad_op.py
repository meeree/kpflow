# Full Hidden NTK Operator. Should be Equivalent to P K P* (see tests). 
import torch
from torch import nn
import numpy as np
from torch.func import vjp, jvp, functional_call
from collections import OrderedDict
from time import perf_counter as pf

from .op_common import Operator

# Utility function
np_to_torch = lambda x: x if torch.is_tensor(x) else torch.from_numpy(x)
torch_to_np = lambda x: x if not torch.is_tensor(x) else x.detach().cpu().numpy()


class FullJThetaOperator(Operator):
    def __init__(self, model, inputs, hidden, dev = 'cpu', params_to_vec = False):
        nparams = sum(p.numel() for _,p in model.named_parameters())
        super().__init__(nparams, hidden.shape, dev = dev, self_adjoint = False, force_shape = True)

        inputs = np_to_torch(inputs).to(dev)
        model_dev = model.to(dev)

        class FnParamsOnly(nn.Module):
            def __init__(self, model, inputs):
                super().__init__()
                self.model = model
                self.inputs = inputs

            def forward(self, params):
                return functional_call(self.model, params, self.inputs)

        self.model = FnParamsOnly(model_dev, inputs)
        self.params = dict(model_dev.named_parameters())
        self.vectorize = params_to_vec # Convert parameters to vectors.
        self.vjp_fn = vjp(self.model, self.params)[1]
        if dev !='cpu':
            self.vjp_fn = torch.compile(self.vjp_fn)

    def vec_to_param(self, vec):
        # Convert a vectorized parameter vec(theta) in R^m to a dictionary real parameter in theta in P.
        out = type(self.params)()  # preserves dict vs OrderedDict
        off = 0
        for k, t in self.params.items():
            n = t.numel()
            out[k] = vec[..., off:off+n].view(*vec.shape[:-1], *t.shape).to(dtype=t.dtype, device=t.device)
            off += n
        if off != vec.shape[-1]:
            raise ValueError(f"Vector length {vec.shape[-1]} != total params {off}")
        return out

    @torch.no_grad()
    def _matvec(self, q):
        # The input should be a dict of named parameters, e.g. dict(model.named_parameters()).
        # Vectorized input for this call is not supported. Only for adjoint_call is this supported for now.
        inp = (q,) if isinstance(q, dict) else q
        if self.vectorize:
            inp = (self.vec_to_param(q),)
        return jvp(self.model, (self.params,), inp)[1]  # [B, T, H]

    @torch.no_grad()
    def _rmatvec(self, q):
        q_torch = np_to_torch(q).to(self.dev)
        vjp_out = self.vjp_fn(q_torch)[0] # A dictionary.
        if not self.vectorize:
            return vjp_out
        return nn.utils.parameters_to_vector(vjp_out.values()) # A vector.

class HiddenNTKOperator(Operator):
    def __init__(self, model, inputs, hidden, dev = 'cpu', params_to_vec = False):
        super().__init__(hidden.shape, hidden.shape, dev, self_adjoint = True, force_shape = True)
        self.jtheta_op = FullJThetaOperator(model, inputs, hidden, dev, params_to_vec = params_to_vec)

    def _matvec(self, q):
        return self.jtheta_op._matvec(self.jtheta_op._rmatvec(q)) # J_theta @ J_theta^T @ q.
