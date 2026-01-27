# Implements the inverse of the propagation operator
import torch
import numpy as np
from torch.func import vjp, jvp, functional_call
from functools import partial

from .lyap_utils import compute_jacobians
from .op_common import Operator

# Utility function
np_to_torch = lambda x: x if torch.is_tensor(x) else torch.from_numpy(x)
torch_to_np = lambda x: x if not torch.is_tensor(x) else x.detach().cpu().numpy()

class FrechetOperator(Operator):
    def __init__(self, model_f, inputs, hidden, dev = 'cpu'):
        super().__init__(hidden.shape, hidden.shape, dev)

        inputs = np_to_torch(inputs).to(dev)
        hidden = np_to_torch(hidden).to(dev)

        inputs_flat = inputs.reshape((-1, inputs.shape[-1]))
        model_f_dev = model_f.to(dev)
        model_f_hidden_only = partial(model_f_dev, inputs_flat)
        self.jacs = compute_jacobians(model_f_hidden_only, hidden, to_np = False) # [B, T, H, H]
        self.jacs_T = self.jacs.swapaxes(-2, -1) # View
        self.time_down = lambda x : torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), 1) # [x(1), x(2), x(3), ..] -> [0, x(1), x(2), ..] in time.
        self.time_up = lambda x : torch.cat((x[:, 1:], torch.zeros_like(x[:, :1])), 1) # [x(1), x(2), x(2), ..] -> [x(2), x(3), x(4), .., 0] in time.

    @torch.no_grad
    def _matvec(self, q):
        q_dev = np_to_torch(q).reshape(self.shape_in)[...,None].float() # Shape [B, T, H, 1].
        q_dev_shift = self.time_down(q_dev)
        res = q_dev - self.jacs @ q_dev_shift
        return res

    @torch.no_grad
    def _rmatvec(self, q):
        q_dev = np_to_torch(q).reshape(self.shape_in)[...,None].float() # Shape [B, T, H, 1].
        q_dev_shift = self.time_up(q_dev)
        res = q_dev - self.jacs_T @ q_dev_shift
        return res

    def __str__(self):
        return f"P.inv{tuple(self.shape_in)}"
