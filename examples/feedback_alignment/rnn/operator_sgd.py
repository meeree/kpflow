"""
Minimal operator-SGD utilities.

This file intentionally keeps the optimizer generic and pushes model-specific
operator construction into small callbacks.  There are two paths:

1. global_constraint_sgd: generic GlobalConstraint path
       update = (D_theta F)^* P^* err_h
   where P = (D_h F)^{-1}.  This is the P/K view without explicitly forming K.

2. weight_based_output_sgd / feedback_alignment_sgd: lightweight callback path
   for weight-site/output operators such as WeightBasedOutputGradient.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

import torch

TensorDict = Dict[str, torch.Tensor]


# ------------------------------------------------------------
# Small utilities
# ------------------------------------------------------------

def sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class StepTimer:
    def __init__(self):
        self.times = defaultdict(list)

    def timeit(self, name: str):
        return _TimerContext(self, name)

    def add(self, name: str, dt: float) -> None:
        self.times[name].append(dt)

    def summary(self):
        out = {}
        for k, vals in self.times.items():
            total = sum(vals)
            out[k] = {"total": total, "mean": total / len(vals), "n": len(vals)}
        return out

    def print_summary(self) -> None:
        print()
        print("Timing summary:")
        print("-" * 72)
        print(f"{'step':35s} {'total (s)':>12s} {'mean (ms)':>12s} {'%':>8s}")
        print("-" * 72)
        totals = {k: sum(v) for k, v in self.times.items()}
        grand_total = sum(totals.values())
        for k, total in sorted(totals.items(), key=lambda kv: -kv[1]):
            mean_ms = 1000.0 * total / len(self.times[k])
            pct = 100.0 * total / grand_total if grand_total > 0 else 0.0
            print(f"{k:35s} {total:12.4f} {mean_ms:12.3f} {pct:8.2f}")
        print("-" * 72)
        print(f"{'TOTAL MEASURED':35s} {grand_total:12.4f}")


class _TimerContext:
    def __init__(self, timer: StepTimer, name: str):
        self.timer = timer
        self.name = name

    def __enter__(self):
        sync_if_cuda()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        sync_if_cuda()
        self.timer.add(self.name, time.perf_counter() - self.t0)


def tree_detach(theta: Mapping[str, torch.Tensor]) -> TensorDict:
    return {k: v.detach().clone() for k, v in theta.items()}


def tree_detach_requires_grad(theta: Mapping[str, torch.Tensor]) -> TensorDict:
    return {k: v.detach().clone().requires_grad_(True) for k, v in theta.items()}


def tree_add_scaled(theta: Mapping[str, torch.Tensor], direction: Mapping[str, torch.Tensor], scale: float) -> TensorDict:
    return {k: theta[k] + scale * direction[k] for k in theta}


def tree_sub_scaled(theta: Mapping[str, torch.Tensor], grad: Mapping[str, torch.Tensor], scale: float) -> TensorDict:
    return {k: theta[k] - scale * grad[k] for k in theta}


def loss_error_signal(loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], h: torch.Tensor, target: torch.Tensor):
    """Return (loss, d loss / d h) without differentiating through the model rollout."""
    h_leaf = h.detach().requires_grad_(True)
    loss = loss_fn(h_leaf, target)
    err_h = torch.autograd.grad(loss, h_leaf, retain_graph=False, create_graph=False)[0]
    return loss.detach(), err_h.detach()


def make_random_feedback_params(
    theta: Mapping[str, torch.Tensor],
    *,
    seed: Optional[int] = None,
    keys: Optional[Iterable[str]] = None,
    match_norm: bool = True,
) -> TensorDict:
    """
    Shape-preserving random feedback weights.

    This preserves whatever blocking/sparsity is represented by separate tensors.
    For example, an RNN with W_h and W_x gets separate random tensors with the
    same shapes.  No dense concatenated matrix is created, so zero/off-block areas
    are not introduced.
    """
    gen = None
    if seed is not None:
        gen = torch.Generator(device=next(iter(theta.values())).device)
        gen.manual_seed(seed)

    keyset = set(keys) if keys is not None else set(theta.keys())
    out = {}
    for k, v in theta.items():
        if k not in keyset:
            out[k] = v.detach().clone()
            continue
        r = torch.randn(v.shape, dtype=v.dtype, device=v.device, generator=gen)
        if match_norm:
            denom = r.norm().clamp_min(1e-12)
            r = r * (v.detach().norm() / denom)
        out[k] = r.detach()
    return out


# ------------------------------------------------------------
# Generic GlobalConstraint SGD: P/K view
# ------------------------------------------------------------

def global_constraint_sgd(
    *,
    theta0: Mapping[str, torch.Tensor],
    batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    lr: float,
    rollout_fn: Callable[[Mapping[str, torch.Tensor], torch.Tensor], torch.Tensor],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    constraint_factory: Callable[[Tuple[Any, ...]], Any],
    primals_factory: Callable[[torch.Tensor, Mapping[str, torch.Tensor], torch.Tensor], Tuple[Any, ...]],
    inv_solver: str = "neumann",
    green_depth: Optional[int] = None,
    print_every: int = 25,
    require_grad_theta: bool = True,
):
    """
    Minimal generic SGD using a GlobalConstraint-like object.

    Required constraint API:
        F.param_jac(primals).adjoint_call(lambda_h)
        F.greens(primals, solver=..., max_iter=..., tol=...).adjoint_call(err_h)

    Math:
        P = (D_h F)^{-1}
        update = (D_theta F)^* P^* err_h

    With residual convention F(h, theta, x)=0, this update is the negative
    parameter gradient direction, so the step is theta += lr * update.
    """
    theta = tree_detach_requires_grad(theta0) if require_grad_theta else tree_detach(theta0)
    losses, runtimes = [], []
    timer = StepTimer()
    sync_if_cuda()
    t0 = time.perf_counter()

    for itr, (x, target) in enumerate(batches):
        with timer.timeit("forward rollout"):
            h = rollout_fn(theta, x)
            primals = primals_factory(h, theta, x)

        with timer.timeit("loss + err_h"):
            loss, err_h = loss_error_signal(loss_fn, h, target)

        with timer.timeit("construct constraint"):
            F = constraint_factory(primals)

        with timer.timeit("construct DthetaF"):
            DthetaF = F.param_jac(primals)

        with timer.timeit("construct Green inverse"):
            max_iter = green_depth if green_depth is not None else h.shape[1] + 1
            P = F.greens(primals, solver=inv_solver, max_iter=max_iter, tol=1e-10)

        with timer.timeit("apply P^*"):
            lambda_h = P.adjoint_call(err_h)

        with timer.timeit("apply DthetaF^*"):
            update = DthetaF.adjoint_call(lambda_h)

        with timer.timeit("parameter update"):
            theta = tree_add_scaled(theta, update, lr)
            theta = tree_detach_requires_grad(theta) if require_grad_theta else tree_detach(theta)

        losses.append(float(loss.item()))
        sync_if_cuda()
        runtimes.append(time.perf_counter() - t0)
        if print_every is not None and itr % print_every == 0:
            print(f"[global ] itr {itr:04d} | loss {loss.item():.6e}")

    return theta, losses, runtimes, timer


# ------------------------------------------------------------
# Weight-site / output-gradient path
# ------------------------------------------------------------

def weight_site_rmatvec(h_site: torch.Tensor, da: torch.Tensor) -> torch.Tensor:
    """(h_site \otimes I)^* da for tensors [..., A] and [..., M]."""
    return torch.einsum("...a,...m->am", da, h_site)


class WeightBasedOutputGradient:
    """
    Generic weight-site output gradient.

    If Dhfo_h_kron_Iy is None, this returns only grad_W/update_W.  If supplied,
    this returns (update_W, update_Wo), matching the two-parameter output case.
    """
    def __init__(self, B, T, O, S, h, Dhfo_h_kron_Iy=None, names=None):
        self.B = B
        self.T = T
        self.O = O
        self.S = S
        self.h = h
        self.Dhfo_h_kron_Iy = Dhfo_h_kron_Iy
        self.names = ("W", "W_o") if names is None else names

    def _rsolve_S(self, b):
        if hasattr(self.S, "rsolve"):
            return self.S.rsolve(b)
        return self.S.T.solve(b)

    def __call__(self, err):
        # z = S^{-*} O^* err
        z = self._rsolve_S(self.O.rmatvec(err))

        # update_W = (h \otimes I_a)^* B^* z under the residual convention.
        update_W = weight_site_rmatvec(self.h, self.B.rmatvec(z))

        if self.Dhfo_h_kron_Iy is None:
            return update_W

        # update_Wo under the same residual convention as the user's sketch.
        update_Wo = -self.Dhfo_h_kron_Iy.rmatvec(err)
        return (update_W, update_Wo)

    def __str__(self):
        return f"OutputGradient[{self.names[0]}, {self.names[1]}]"


def weight_based_output_sgd(
    *,
    theta0: Mapping[str, torch.Tensor],
    batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    lr: float,
    rollout_fn: Callable[[Mapping[str, torch.Tensor], torch.Tensor], torch.Tensor],
    loss_error_fn: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    ops_builder: Callable[..., WeightBasedOutputGradient],
    unpack_update_fn: Callable[[Any, Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]],
    feedback_params: Optional[Mapping[str, torch.Tensor]] = None,
    print_every: int = 25,
    label: str = "wbo",
):
    """
    Minimal SGD loop for WeightBasedOutputGradient-like operator builders.

    ops_builder(theta=theta, x=x, h=h, feedback_theta=feedback_params) should
    construct the backward operators for the current forward state.  For vanilla
    SGD, feedback_params=None.  For FA, feedback_params is a fixed random copy.
    """
    theta = tree_detach(theta0)
    feedback_params = None if feedback_params is None else tree_detach(feedback_params)
    losses, runtimes = [], []
    timer = StepTimer()
    sync_if_cuda()
    t0 = time.perf_counter()

    with torch.no_grad():
        for itr, (x, target) in enumerate(batches):
            with timer.timeit("forward rollout"):
                h = rollout_fn(theta, x).detach()

            with timer.timeit("loss + err"):
                loss, err = loss_error_fn(h, target)

            with timer.timeit("construct backprop operators"):
                grad_op = ops_builder(theta=theta, x=x, h=h, feedback_theta=feedback_params)

            with timer.timeit("apply backprop operators"):
                raw_update = grad_op(err)
                update = unpack_update_fn(raw_update, theta)

            with timer.timeit("parameter update"):
                theta = tree_add_scaled(theta, update, lr)
                theta = tree_detach(theta)

            losses.append(float(loss.item()))
            sync_if_cuda()
            runtimes.append(time.perf_counter() - t0)
            if print_every is not None and itr % print_every == 0:
                print(f"[{label:7s}] itr {itr:04d} | loss {loss.item():.6e}")

    return theta, losses, runtimes, timer


def feedback_alignment_sgd(
    *,
    theta0: Mapping[str, torch.Tensor],
    batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    lr: float,
    rollout_fn: Callable[[Mapping[str, torch.Tensor], torch.Tensor], torch.Tensor],
    loss_error_fn: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    ops_builder: Callable[..., WeightBasedOutputGradient],
    unpack_update_fn: Callable[[Any, Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]],
    feedback_seed: int = 0,
    feedback_keys: Optional[Iterable[str]] = None,
    match_feedback_norm: bool = True,
    print_every: int = 25,
):
    """
    Feedback-alignment variant for the WeightBasedOutputGradient path.

    The forward pass and updated parameters use theta.  The backward operators
    receive a fixed random feedback_theta with the same tensor blocks as theta.
    The ops_builder decides which feedback blocks are used.  For the RNN demo,
    only W_h.weight enters the hidden-state backpropagator T^*; W_x.weight is
    randomized too but does not create a forbidden dense off-block connection.
    """
    feedback_params = make_random_feedback_params(
        theta0,
        seed=feedback_seed,
        keys=feedback_keys,
        match_norm=match_feedback_norm,
    )
    return weight_based_output_sgd(
        theta0=theta0,
        batches=batches,
        lr=lr,
        rollout_fn=rollout_fn,
        loss_error_fn=loss_error_fn,
        ops_builder=ops_builder,
        unpack_update_fn=unpack_update_fn,
        feedback_params=feedback_params,
        print_every=print_every,
        label="fa",
    )


# ------------------------------------------------------------
# Plain PyTorch baseline
# ------------------------------------------------------------

def pytorch_direct_sgd(
    *,
    theta0: Mapping[str, torch.Tensor],
    batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    lr: float,
    rollout_fn: Callable[[Mapping[str, torch.Tensor], torch.Tensor], torch.Tensor],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    print_every: int = 25,
):
    theta = tree_detach_requires_grad(theta0)
    losses, runtimes = [], []
    sync_if_cuda()
    t0 = time.perf_counter()

    for itr, (x, target) in enumerate(batches):
        theta = tree_detach_requires_grad(theta)
        h = rollout_fn(theta, x)
        loss = loss_fn(h, target)
        grad_values = torch.autograd.grad(loss, tuple(theta.values()))
        grad = {name: g.detach() for (name, _), g in zip(theta.items(), grad_values)}
        theta = tree_sub_scaled(theta, grad, lr)
        theta = tree_detach_requires_grad(theta)

        losses.append(float(loss.detach().item()))
        sync_if_cuda()
        runtimes.append(time.perf_counter() - t0)
        if print_every is not None and itr % print_every == 0:
            print(f"[pytorch] itr {itr:04d} | loss {loss.item():.6e}")

    return theta, losses, runtimes
