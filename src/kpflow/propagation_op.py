# Propagator operator from main paper. 
import torch
import numpy as np
from torch.func import vjp, jvp, functional_call
from functools import partial

from .lyap_utils import compute_jacobians
from .op_common import Operator

# Utility function
np_to_torch = lambda x: x if torch.is_tensor(x) else torch.from_numpy(x)
torch_to_np = lambda x: x if not torch.is_tensor(x) else x.detach().cpu().numpy()

class PropagationOperator_LinearForm(Operator):
    def __init__(self, model_f, inputs, hidden, dev = 'cpu'):
        super().__init__(hidden.shape, hidden.shape, dev)

        inputs = np_to_torch(inputs).to(dev)
        hidden = np_to_torch(hidden).to(dev)

        inputs_flat = inputs.reshape((-1, inputs.shape[-1]))
        model_f_dev = model_f.to(dev)
        model_f_hidden_only = partial(model_f_dev, inputs_flat)
        self.jacs = compute_jacobians(model_f_hidden_only, hidden, to_np = False) # [B, T, H, H]
        self.jacs_T = self.jacs.swapaxes(-2, -1) # View
#        self.Qs, self.Rs, self.Us = fundamental_matrix(self.jacs)

    @torch.no_grad
    def _matvec(self, q):
        q_dev = np_to_torch(q).reshape(self.shape_in)[...,None].float() # Shape [B, T, H, 1].
        res = [torch.zeros_like(q_dev[:, 0])]
        for t in range(q_dev.shape[1]):
            res.append(q_dev[:, t] + self.jacs[:, t] @ res[-1])

        return torch.stack(res[1:], 1)[...,0]

    @torch.no_grad
    def _rmatvec(self, q):
        q_dev = np_to_torch(q).reshape(self.shape_out)[...,None].float() # Shape [B, T, H, 1].
        res = [q_dev[:, -1]]
        for t in range(q_dev.shape[1]-2, -1, -1):
            res.append(q_dev[:, t] + self.jacs_T[:, t+1] @ res[-1])

        return torch.stack(res, 1)[...,0].flip(1) # Reverse time.

    def __str__(self):
        return f"P{tuple(self.shape_in)}"

## Implements P P^T in a stable efficient way.
#class PropagationGramOperator(Operator):
#    def __init__(self, model_f, inputs, hidden, dev = 'cpu'):
#        super().__init__()
#        self.dev = dev
#
#        inputs = np_to_torch(inputs).to(dev)
#        hidden = np_to_torch(hidden).to(dev)
#
#        inputs_flat = inputs.reshape((-1, inputs.shape[-1]))
#        hidden_flat = hidden.reshape((-1, hidden.shape[-1]))
#        model_f = partial(model_f, inputs_flat)
#        jacs = compute_jacobians(model_f, hidden, to_np = False) # [B, T, H, H]
#        self.Qs, self.Rs = fundamental_matrix(jacs, return_U = False)
#
#        self.self_adjoint = True # P P^T is self adjoint and positive definite.
#
#        self.V = lambda q: torch.cumsum(q, axis = 1) # Volterra integration operator.
#        self.Vt = lambda q: torch.cumsum(q.flip(1), axis = 1).flip(1) # Reverse time volterra operator.
#    @torch.no_grad
#    def __call__(self, q):
#        # q is shape [..., B, T, H].
#        # Three step process:
#        # (1) Solve R^T q1 = (q^T (Q R V))^T 
#        # (2) Solve R q2 = q1
#        # (3) Let q3 = (Q R V) q2.
#        q_dev = np_to_torch(q).to(self.dev).float()
#        if len(q.shape) > 3:
#            q_dev = q_dev.reshape((-1, *q_dev.shape[-3:])).moveaxis(0, -1) # [..., B, T, H] -> [B, T, H, k]. 
#        else:
#            q_dev = q_dev[..., None] # [B, T, H] -> [B, T, H, 1].
#
#        rhs = self.Vt(self.Rs.swapaxes(-2,-1) @ self.Qs.swapaxes(-2,-1) @ q_dev)
##        q1 = torch.linalg.solve(self.Rs.swapaxes(-2,-1), rhs)
##        q2 = torch.linalg.solve(self.Rs, q1)
#        q1 = torch.linalg.lstsq(self.Rs.swapaxes(-2,-1), rhs)[0]
#        q2 = torch.linalg.lstsq(self.Rs, q1)[0]
#        q3 = self.Qs @ self.Rs @ self.V(q2)
#        if q_dev.shape[0] == 1 and self.Qs.shape[0] != 1: # If B = 1 is inputted, take a mean along batch dim.
#            q3 = q3.mean(0)
#        return q3.moveaxis(-1, 0).reshape(q.shape) # [B, T, H, k] -> [..., B, T, H], original shape.

class PropagationOperator_DirectForm(Operator):
    def __init__(self, model_f, x, hidden, h0 = None, dev = 'cpu'):
        super().__init__(hidden.shape, hidden.shape, dev)
        self.model_f = model_f.to(dev)
        self.x = np_to_torch(x).to(dev) # [B, T, Nin]
        self.h0 = None if h0 is None else np_to_torch(h0).to(dev)

        # Evaluation with no perturbations to model.
        self.baseline_eval = self.evaluate_model(torch.zeros(hidden.shape).to(dev))
        for h in self.baseline_eval:
            h.requires_grad_()
            h.retain_grad()

        self.batched_call = self._matvec
        self.batched_adjoint_call = self._rmatvec

    # Evaluate model(f + q) where f is the original model, q is a perturbation to the forcing term.
    def evaluate_model(self, q_dev):
#        res = []
#        for t in range(q_dev.shape[1]):
#            h = self.model_f(self.x[:, t], h) + q_dev[:, t]
#            res.append(h.clone())
        h0 = self.h0.clone() if self.h0 is not None else torch.zeros_like(q_dev[:, 0])
        hidden = [h0]
        hidden[0] = hidden[0].to(self.dev)
        for t in range(self.x.shape[1]):
            hidden.append(self.model_f(self.x[:, t], hidden[-1]) + q_dev[:, t])
        hidden = hidden[1:]
        return hidden
        for h in hidden:
            h.requires_grad_()
            h.retain_grad()
        return res # A list

    @torch.no_grad
    def _matvec(self, q):
        # q is shape (..., B, T, H) where ... can be any dimension.
        # Evaluate forwards(model_f + q) - forwards(model_f).
        q_dev = np_to_torch(q).to(self.dev).float() # [B, T, H]
        eval_with_perb = self.evaluate_model(q_dev)
        res = torch.stack(eval_with_perb, -2) - torch.stack(self.baseline_eval, -2)
        return res

    def _rmatvec(self, q):
        # Adjoint maps grad(l(z(t|x)), z(t|x)) to grad(Loss, z(t|x)) where grad(mean(l(z(t|x))), z(t|x)).
        # So, if we let q(t|x) = grad(l(z(t|x)), z(t|x)), we get the output. To do this, we can let
        # l(z(t|x)) = <z(t|x), q(t|x)>.
#        for h in self.baseline_eval:
#            if h.grad is not None:
#                h.grad.zero_() # Zero out previous grads in there.

        hidden = self.evaluate_model(torch.zeros(*self.x.shape[:-1], q.shape[-1]).to(self.dev))
        for h in hidden:
            h.requires_grad_()
            h.retain_grad()

        q_dev = np_to_torch(q).to(self.dev).float() # [..., B, T, H] = [k1, ..., kn, B, T, H] where k1, ..., kn are optional.
        B, T = q_dev.shape[-3:-1]
        total = sum([q_dev[:,t] * hidden[t] for t in range(T)]) 
        surrogate_loss = total.sum((-2, -1)) 
        surrogate_loss.backward(retain_graph=True)
        res = torch.stack([h.grad for h in hidden], 1)
        return res

#    # Evaluate model(f + q) where f is the original model, q is a perturbation to the forcing term.
#    def evaluate_model(self, q):
#        q_dev = np_to_torch(q).to(self.dev) # [..., B, T, H] = [k1, ..., kn, B, T, H] where k1, ..., kn are optional.
#        q_dev = q_dev.reshape((-1, *q_dev.shape[-2:])) # [k1*...*kn*B, T, H], i.e. combine into one batch dim.
#        x_in = self.x
#        if q_dev.shape[0] > x_in.shape[0]:
#            ndup = q_dev.shape[0] // x_in.shape[0]
#            x_in = torch.cat([x_in] * ndup, 0)
#
#        h = self.h0.clone() if self.h0 is not None else torch.zeros_like(q_dev[:, 0])
#        res = []
#        for t in range(q_dev.shape[1]):
#            h = self.model_f(x_in[:, t], h) + q_dev[:, t]
#            res.append(h.clone().reshape(q[..., 0, :].shape))
#        return res # A list
#
#    @torch.no_grad
#    def __call__(self, q):
#        # q is shape (..., B, T, H) where ... can be any dimension.
#        # Evaluate forwards(model_f + q) - forwards(model_f).
#        eval_with_perb = self.evaluate_model(q)
#        return torch_to_np(torch.stack(eval_with_perb, -2) - torch.stack(self.baseline_eval, -2))
#
#    def adjoint_call(self, q):
#        # Adjoint maps grad(l(z(t|x)), z(t|x)) to grad(Loss, z(t|x)) where grad(mean(l(z(t|x))), z(t|x)).
#        # So, if we let q(t|x) = grad(l(z(t|x)), z(t|x)), we get the output. To do this, we can let
#        # l(z(t|x)) = <z(t|x), q(t|x)>.
#        for h in self.baseline_eval:
#            if h.grad is not None:
#                h.grad.zero_() # Zero out previous grads in there.
#
#        q_dev = np_to_torch(q).to(self.dev) # [..., B, T, H] = [k1, ..., kn, B, T, H] where k1, ..., kn are optional.
#        B, T = q_dev.shape[-3:-1]
#
#        if len(q_dev.shape) == 3:
#            total = sum([q_dev[:,t] * self.baseline_eval[t] / T for t in range(T)]) 
#            surrogate_loss = total.sum((-2, -1)) / B
#            surrogate_loss.backward(retain_graph=True)
#            return torch_to_np(torch.stack([h.grad for h in self.baseline_eval], 1))
#
#        
#        # Assume q_dev is shame [k, B, T, H].
#        hidden_stack = [self.baseline_eval] * q_dev.shape[0]
#        total = sum([sum([q_dev[k,:,t] * hidden_stack[k][t] / T for t in range(T)]) for k in range(q_dev.shape[0])])
#        surrogate_loss = total.sum((-2, -1)) / B
#        surrogate_loss.backward(retain_graph=True)
#        return torch_to_np(torch.stack([torch.stack([h.grad for h in hidden], 1) for hidden in hidden_stack])) # Inner stack is [B, T, H]. Outer stack becomes [k, B, T, H].

#
#        from tqdm import tqdm
#        res = []
#        for t in tqdm(range(q.shape[1])):
#            res.append(torch.zeros_like(q_dev[:, 0])) # [B, H, 1].
#            for t0 in range(t+1):
#                res[-1] += state_transition(self.Qs, self.Rs, t, t0) @ q_dev[:, t0]
#        return torch_to_np(torch.stack(res, 1)[...,0])
#
#        from tqdm import tqdm
#        res = []
#        for t in tqdm(range(q.shape[1])):
#            res.append(torch.zeros_like(q_dev[:, 0])) # [B, H, 1].
#            if t < q.shape[1] - 2:
#                continue
#            print(t, q.shape[1])
#            for t0 in range(t, q.shape[1]):
#                res[-1] += state_transition(self.Qs, self.Rs, t0, t).swapaxes(-2,-1) @ q_dev[:, t0]
#        return torch_to_np(torch.stack(res, 1)[...,0])
