"""
RNN demo comparing:
  1. PyTorch direct SGD
  2. Weight-based operator SGD
  3. Weight-based feedback alignment
  4. Optional GlobalConstraint operator SGD, if kpflow is importable

Run from the same directory as operator_sgd.py:
    python demo_rnn_fa_vs_sgd.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.func import functional_call

from operator_sgd import (
    WeightBasedOutputGradient,
    feedback_alignment_sgd,
    global_constraint_sgd,
    pytorch_direct_sgd,
    tree_detach,
    weight_based_output_sgd,
)

try:
    from kpflow.implicit import GlobalConstraint
except Exception:  # keep this demo runnable without kpflow installed
    GlobalConstraint = None


# ------------------------------------------------------------
# RNN + data
# ------------------------------------------------------------

def step_down(h):
    return torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)


def step_up(q):
    return torch.cat([q[:, 1:], torch.zeros_like(q[:, :1])], dim=1)


class TinyRNNCell(nn.Module):
    def __init__(self, n_in, n_hidden):
        super().__init__()
        self.W_h = nn.Linear(n_hidden, n_hidden, bias=False)
        self.W_x = nn.Linear(n_in, n_hidden, bias=False)

    def forward(self, h_prev, x):
        return torch.tanh(self.W_h(h_prev) + self.W_x(x))


def rollout_torch(model, theta, x):
    B, T, _ = x.shape
    N = theta["W_h.weight"].shape[0]
    h_prev = torch.zeros(B, N, dtype=x.dtype, device=x.device)
    hs = []
    for t in range(T):
        h_prev = functional_call(model, theta, (h_prev, x[:, t]))
        hs.append(h_prev)
    return torch.stack(hs, dim=1)


def rollout_weight(theta, x):
    W_h = theta["W_h.weight"]
    W_x = theta["W_x.weight"]
    B, T, _ = x.shape
    N = W_h.shape[0]
    h_prev = torch.zeros(B, N, dtype=x.dtype, device=x.device)
    hs = []
    for t in range(T):
        h_prev = torch.tanh(h_prev @ W_h.T + x[:, t] @ W_x.T)
        hs.append(h_prev)
    return torch.stack(hs, dim=1)


def make_fixed_teacher_batches(teacher_model, teacher_theta, nitr, B, T, N_in):
    batches = []
    for _ in range(nitr):
        x = torch.randn(B, T, N_in)
        with torch.no_grad():
            target = rollout_torch(teacher_model, teacher_theta, x)
        batches.append((x, target))
    return batches


def mse_loss(h, target):
    return 0.5 * ((h - target) ** 2).mean()


def mse_loss_and_error(h, target):
    diff = h - target
    return 0.5 * (diff ** 2).mean(), diff / h.numel()


# ------------------------------------------------------------
# Weight-based backprop operators for this RNN
# ------------------------------------------------------------

class ElementwiseBOperator:
    """B = D_a(h - tanh(a)) = -diag(1 - h^2)."""
    def __init__(self, h):
        self.dphi = 1.0 - h ** 2

    def __call__(self, v):
        return -self.dphi * v

    def rmatvec(self, w):
        return -self.dphi * w


class WeightSiteTOperator:
    """T dh = step_down(dh) W_h^T."""
    def __init__(self, W_h):
        self.W_h = W_h

    def __call__(self, v):
        return step_down(v) @ self.W_h.T

    def rmatvec(self, w):
        return step_up(w @ self.W_h)


class IdentityOperator:
    def __call__(self, v):
        return v

    def rmatvec(self, w):
        return w


class SOperator:
    """S = I + B T, with exact finite Neumann solve for the causal RNN."""
    def __init__(self, B, T, max_iter):
        self.B = B
        self.T = T
        self.max_iter = max_iter

    def __call__(self, v):
        return v + self.B(self.T(v))

    def rmatvec(self, w):
        return w + self.T.rmatvec(self.B.rmatvec(w))

    def solve(self, b):
        z = b
        term = b
        for _ in range(self.max_iter):
            term = -self.B(self.T(term))
            z = z + term
        return z

    def rsolve(self, b):
        z = b
        term = b
        for _ in range(self.max_iter):
            term = -self.T.rmatvec(self.B.rmatvec(term))
            z = z + term
        return z


def make_rnn_wbo_ops(*, theta, x, h, feedback_theta=None, green_depth=None):
    """
    Build the backward operators for vanilla BP or FA.

    Vanilla: feedback_theta is None, so W_h_back = theta["W_h.weight"].
    FA:      W_h_back is fixed random feedback_theta["W_h.weight"].

    The forward weight sites h_site = [h_{t-1}, x_t] stay real.  This means the
    update still has the correct W_h / W_x block structure and does not create
    off-block weights.
    """
    Bop = ElementwiseBOperator(h)
    W_h_back = theta["W_h.weight"] if feedback_theta is None else feedback_theta["W_h.weight"]
    Top = WeightSiteTOperator(W_h_back)
    Oop = IdentityOperator()
    Sop = SOperator(Bop, Top, max_iter=green_depth if green_depth is not None else h.shape[1] + 1)
    h_site = torch.cat([step_down(h), x], dim=-1)
    return WeightBasedOutputGradient(B=Bop, T=Top, O=Oop, S=Sop, h=h_site)


def unpack_rnn_update(update_cat, theta):
    N = theta["W_h.weight"].shape[1]
    return {
        "W_h.weight": update_cat[:, :N],
        "W_x.weight": update_cat[:, N:],
    }


# ------------------------------------------------------------
# Optional GlobalConstraint setup
# ------------------------------------------------------------

def implicit_F(model):
    def F(h, theta, x):
        h_prev = step_down(h)
        pred = functional_call(model, theta, (h_prev, x))
        return h - pred
    return lambda tpl: F(*tpl)


def make_global_constraint_factory(model):
    if GlobalConstraint is None:
        raise RuntimeError("kpflow.implicit.GlobalConstraint is not importable.")

    def factory(primals):
        return GlobalConstraint(implicit_F(model), primals, state_idx=0, param_idx=1)

    return factory


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    torch.manual_seed(0)

    B = 32
    T = 12
    N_in = 4
    N = 8
    nitr = 9000
    lr = 5e-2
    print_every = 100

    teacher = TinyRNNCell(N_in, N)
    with torch.no_grad():
        teacher.W_h.weight.mul_(0.5)
        teacher.W_x.weight.mul_(0.8)
    teacher_theta = {name: p.detach().clone() for name, p in teacher.named_parameters()}

    student = TinyRNNCell(N_in, N)
    with torch.no_grad():
        student.W_h.weight.mul_(0.2)
        student.W_x.weight.mul_(0.2)
    theta0 = {name: p.detach().clone() for name, p in student.named_parameters()}

    torch.manual_seed(123)
    batches = make_fixed_teacher_batches(teacher, teacher_theta, nitr, B, T, N_in)

    print("\nRunning PyTorch direct SGD...")
    theta_torch, losses_torch, times_torch = pytorch_direct_sgd(
        theta0=theta0,
        batches=batches,
        lr=lr,
        rollout_fn=lambda th, xx: rollout_torch(student, th, xx),
        loss_fn=mse_loss,
        print_every=print_every,
    )

    print("\nRunning weight-based operator SGD...")
    theta_wbo, losses_wbo, times_wbo, _ = weight_based_output_sgd(
        theta0=theta0,
        batches=batches,
        lr=lr,
        rollout_fn=rollout_weight,
        loss_error_fn=mse_loss_and_error,
        ops_builder=lambda **kw: make_rnn_wbo_ops(**kw, green_depth=T + 1),
        unpack_update_fn=unpack_rnn_update,
        print_every=print_every,
        label="wbo",
    )

    print("\nRunning feedback alignment...")
    theta_fa, losses_fa, times_fa, _ = feedback_alignment_sgd(
        theta0=theta0,
        batches=batches,
        lr=lr,
        rollout_fn=rollout_weight,
        loss_error_fn=mse_loss_and_error,
        ops_builder=lambda **kw: make_rnn_wbo_ops(**kw, green_depth=T + 1),
        unpack_update_fn=unpack_rnn_update,
        feedback_seed=999,
        feedback_keys=("W_h.weight", "W_x.weight"),
        match_feedback_norm=True,
        print_every=print_every,
    )

    losses_global = times_global = theta_global = None
    if GlobalConstraint is not None:
        print("\nRunning optional GlobalConstraint operator SGD...")
        theta_global, losses_global, times_global, _ = global_constraint_sgd(
            theta0=theta0,
            batches=batches,
            lr=lr,
            rollout_fn=lambda th, xx: rollout_torch(student, th, xx),
            loss_fn=mse_loss,
            constraint_factory=make_global_constraint_factory(student),
            primals_factory=lambda h, th, xx: (h, th, xx),
            inv_solver="neumann",
            green_depth=T + 1,
            print_every=print_every,
        )
    else:
        print("\nSkipping GlobalConstraint run because kpflow is not importable.")

    print("\nFinal losses")
    print(f"PyTorch direct: {losses_torch[-1]:.6e}")
    print(f"WBO operator:   {losses_wbo[-1]:.6e}")
    print(f"FA operator:    {losses_fa[-1]:.6e}")
    if losses_global is not None:
        print(f"Global op:      {losses_global[-1]:.6e}")
        print("Max |PyTorch - Global|:", max(abs(a - b) for a, b in zip(losses_torch, losses_global)))
    print("Max |PyTorch - WBO|:   ", max(abs(a - b) for a, b in zip(losses_torch, losses_wbo)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(losses_torch, label="PyTorch direct")
    axes[0].plot(losses_wbo, "--", label="WBO operator")
    axes[0].plot(losses_fa, ":", label="FA operator")
    if losses_global is not None:
        axes[0].plot(losses_global, "-.", label="GlobalConstraint")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss vs iteration")
    axes[0].set_yscale("log")
    axes[0].legend()

    axes[1].plot(times_torch, losses_torch, label="PyTorch direct")
    axes[1].plot(times_wbo, losses_wbo, "--", label="WBO operator")
    axes[1].plot(times_fa, losses_fa, ":", label="FA operator")
    if times_global is not None:
        axes[1].plot(times_global, losses_global, "-.", label="GlobalConstraint")
    axes[1].set_xlabel("Runtime (seconds)")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss vs runtime")
    axes[1].set_yscale("log")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return {
        "theta_torch": theta_torch,
        "theta_wbo": theta_wbo,
        "theta_fa": theta_fa,
        "theta_global": theta_global,
        "losses_torch": losses_torch,
        "losses_wbo": losses_wbo,
        "losses_fa": losses_fa,
        "losses_global": losses_global,
        "times_torch": times_torch,
        "times_wbo": times_wbo,
        "times_fa": times_fa,
        "times_global": times_global,
    }


if __name__ == "__main__":
    out = main()
