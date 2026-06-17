#!/usr/bin/env python3
"""
Minimal memory-pro / delay-response example for Duhamel split training.

Task:
    A 2D cue is shown at t=0. After a delay, the network must output the
    same 2D direction in h_T[:2]. This is a stripped-down "memory pro"
    task in the spirit of Driscoll-style delayed response tasks.

Model:
    h_{t+1} = W tanh(h_t) + W_in x_{t+1} + b

Split:
    W tanh(h) = W h + W (tanh(h) - h)

We train the same task two ways:

  1. --method linear_base
       Exact linear Green's function for W h,
       truncate the nonlinear correction W(tanh(h)-h).

  2. --method residual_base
       Exact residual/nonlinear Green's function for W(tanh(h)-h),
       truncate the linear correction W h.

Both are implemented with the kpflow-style operator objects:
    LinearRNNStateJacobian, DuhamelGreen, TruncatedNeumann.

Run:
    python memory_pro_split_rnn.py --out figures_memory_pro

Quick:
    python memory_pro_split_rnn.py --quick --out figures_quick

If residual_base struggles on long delays, either shorten --T or increase
--residual-insertions. With one linear insertion it is intentionally a harsher
credit rule than linear_base.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from kpflow.op_common import LinearOperator

from dbptt_compare_operators_fastbase import (
    DuhamelGreen,
    LinearRNNStateJacobian,
    TruncatedNeumann,
    IdentityState,
    make_W,
    spectral_clip_,
    shift_up,
    step_down,
)

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# -----------------------------
# task
# -----------------------------

def make_memory_pro_batch(batch, T, input_dim, device):
    """Cue at t=0, remember until final time."""
    theta = 2.0 * np.pi * torch.rand(batch, device=device)
    cue = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    x = torch.zeros(batch, T, input_dim, device=device)
    x[:, 0, :2] = cue

    # Optional go channel at final time. It is useful for plotting/extension,
    # but the default loss simply reads h_T[:2].
    if input_dim > 2:
        x[:, -1, 2] = 1.0

    target = cue
    h0 = torch.zeros(batch, args_n_placeholder(), device=device)  # overwritten below
    return x, target, cue


def args_n_placeholder():
    # Avoid passing n through make_memory_pro_batch just for h0;
    # train() creates h0 correctly.
    return 1


def make_fixed_dataset(args):
    g = torch.Generator(device=args.device)
    g.manual_seed(args.seed + 123)
    theta = 2.0 * np.pi * torch.rand(args.batch, generator=g, device=args.device)
    cue = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    x = torch.zeros(args.batch, args.T, args.input_dim, device=args.device)
    x[:, 0, :2] = args.cue_scale * cue
    if args.input_dim > 2:
        x[:, -1, 2] = 1.0

    target = cue
    h0 = torch.zeros(args.batch, args.n, device=args.device)
    return h0, x, target


# -----------------------------
# model
# -----------------------------

def rollout(W, Win, b, h0, x):
    """h_{t+1}=W tanh(h_t)+Win x_{t+1}+b. Returns h[0:T]."""
    h = h0
    hs = []
    for t in range(x.shape[1]):
        h = torch.tanh(h) @ W.T + x[:, t] @ Win.T + b
        hs.append(h)
    return torch.stack(hs, dim=1)


def output_loss_and_error(h, target):
    pred = h[:, -1, :2]
    err = pred - target
    loss = 0.5 * err.square().mean()
    e = torch.zeros_like(h)
    e[:, -1, :2] = err / h.shape[0]
    rmse = err.square().mean().sqrt()
    cos = ((pred * target).sum(-1) / (pred.norm(dim=-1) * target.norm(dim=-1) + 1e-8)).mean()
    return loss, e, pred, rmse, cos


# -----------------------------
# split operators for W tanh(h) = W h + W(tanh(h)-h)
# -----------------------------

class TanhResidualStateJacobian(LinearOperator):
    """D_hF contribution from the residual term -W(tanh(h)-h).

    Full constraint:
        F_t = h_t - W h_{t-1} - W(tanh(h_{t-1}) - h_{t-1}) - input_t.

    This operator is only the residual contribution:
        C(dh)_t = - W diag(tanh'(h_{t-1}) - 1) dh_{t-1}.
    In row-vector code:
        C(dh)_t = - ((dh_{t-1} * slope_t) @ W.T)
    where slope_t = tanh'(h_{t-1}) - 1.
    """

    def __init__(self, h, h0, W):
        self.W = W
        prev_h = step_down(h, h0)
        self.slope = 1.0 - torch.tanh(prev_h).square() - 1.0
        super().__init__(h.shape, h.shape, dev=h.device)

    def _matvec(self, dh):
        return -((step_down(dh) * self.slope) @ self.W.T)

    def _rmatvec(self, w):
        return -((shift_up(w) @ self.W) * shift_up(self.slope))

    def __str__(self):
        return "D_hF[tanh residual]"


class TanhResidualBaseStateJacobian(LinearOperator):
    """Base constraint Jacobian for residual-only dynamics.

    S_res(dh) = dh - W diag(tanh'(h_prev)-1) dh_prev.

    This is I plus the residual contribution above, and it has an exact
    triangular inverse just like the linear RNN operator.
    """

    def __init__(self, h, h0, W):
        self.W = W
        prev_h = step_down(h, h0)
        self.slope = 1.0 - torch.tanh(prev_h).square() - 1.0
        super().__init__(h.shape, h.shape, dev=h.device)

    def _matvec(self, dh):
        return dh - ((step_down(dh) * self.slope) @ self.W.T)

    def _rmatvec(self, w):
        return w - ((shift_up(w) @ self.W) * shift_up(self.slope))

    def inverse(self, solver="neumann", solver_kwargs=None, **kwargs):
        return TanhResidualGreen(self.shape_in, self.W, self.slope, self.dev)

    def __str__(self):
        return "D_hF[residual-base]"


class TanhResidualGreen(LinearOperator):
    """Exact inverse for TanhResidualBaseStateJacobian."""

    def __init__(self, shape, W, slope, device):
        self.W = W
        self.slope = slope
        super().__init__(shape, shape, dev=device)

    def _matvec(self, b):
        B, T, N = b.shape
        ys = []
        prev = torch.zeros((B, N), dtype=b.dtype, device=b.device)
        WT = self.W.T
        for t in range(T):
            prev = b[:, t] + (prev * self.slope[:, t]) @ WT
            ys.append(prev)
        return torch.stack(ys, dim=1)

    def _rmatvec(self, b):
        B, T, N = b.shape
        ys = [None] * T
        nxt = torch.zeros((B, N), dtype=b.dtype, device=b.device)
        W = self.W
        for t in range(T - 1, -1, -1):
            if t == T - 1:
                nxt = b[:, t]
            else:
                nxt = b[:, t] + (nxt @ W) * self.slope[:, t + 1]
            ys[t] = nxt
        return torch.stack(ys, dim=1)

    def __str__(self):
        return "Green[D_hF residual exact]"


def make_green(h, h0, W, method, insertions):
    """Return approximate Green's operator P_k for the chosen split."""
    S_linear = LinearRNNStateJacobian(h, W)
    S_residual_piece = TanhResidualStateJacobian(h, h0, W)
    S_full = S_linear + S_residual_piece

    if method == "direct":
        return TruncatedNeumann(S_full, insertions)

    if method == "linear_base":
        # Exact W h propagation; truncate W(tanh-h) insertions.
        P_base = S_linear.inverse(solver="neumann", solver_kwargs={"max_iter": h.shape[1] + 1})
        perturbation = S_linear - S_full
        return DuhamelGreen(P_base, perturbation, insertions + 1)

    if method == "residual_base":
        # Exact W(tanh-h) residual propagation; truncate W h insertions.
        S_res_base = TanhResidualBaseStateJacobian(h, h0, W)
        P_base = S_res_base.inverse(solver="neumann", solver_kwargs={"max_iter": h.shape[1] + 1})
        perturbation = S_res_base - S_full
        return DuhamelGreen(P_base, perturbation, insertions + 1)

    raise ValueError(f"unknown method {method}")


# -----------------------------
# manual implicit gradients
# -----------------------------

def param_grads_from_credit(credit, h, h0, x):
    """True dL/dtheta from credit=P.T(dL/dh)."""
    prev_h = step_down(h, h0)
    phi = torch.tanh(prev_h)

    # F = h - W tanh(prev_h) - Win x - b
    # DthetaF.T credit is negative. True grad is -DthetaF.T credit.
    grad_W = torch.einsum("btn,btm->nm", credit, phi)
    grad_Win = torch.einsum("btn,bti->ni", credit, x)
    grad_b = credit.sum(dim=(0, 1))
    return grad_W, grad_Win, grad_b


def train_one(args, method, insertions):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    W = make_W(args.n, args.lambda_w, args.bulk_radius, rng, args.device)
    W = args.w_scale * W
    Win = args.win_scale * torch.randn(args.n, args.input_dim, device=args.device)
    b = torch.zeros(args.n, device=args.device)

    h0, x, target = make_fixed_dataset(args)

    hist = {
        "step": [],
        "loss": [],
        "rmse": [],
        "cos": [],
        "rho": [],
        "grad_W": [],
        "grad_Win": [],
    }

    pbar = tqdm(range(args.steps + 1), desc=method, dynamic_ncols=True)
    for step in pbar:
        with torch.no_grad():
            h = rollout(W, Win, b, h0, x)
            loss, e, pred, rmse, cos = output_loss_and_error(h, target)
            P = make_green(h, h0, W, method, insertions)
            credit = P.T(e)
            gW, gWin, gb = param_grads_from_credit(credit, h, h0, x)

            if step % args.log_every == 0 or step == args.steps:
                hist["step"].append(step)
                hist["loss"].append(float(loss.cpu()))
                hist["rmse"].append(float(rmse.cpu()))
                hist["cos"].append(float(cos.cpu()))
                hist["rho"].append(float(torch.linalg.eigvals(W).abs().max().real.cpu()))
                hist["grad_W"].append(float(gW.norm().cpu()))
                hist["grad_Win"].append(float(gWin.norm().cpu()))
                pbar.set_postfix(loss=f"{hist['loss'][-1]:.3e}", rmse=f"{hist['rmse'][-1]:.3f}")

            if step == args.steps:
                break

            if gW.norm() > args.grad_clip:
                gW = gW * (args.grad_clip / (gW.norm() + 1e-12))
            if gWin.norm() > args.grad_clip:
                gWin = gWin * (args.grad_clip / (gWin.norm() + 1e-12))
            if gb.norm() > args.grad_clip:
                gb = gb * (args.grad_clip / (gb.norm() + 1e-12))

            W -= args.lr_W * gW
            Win -= args.lr_Win * gWin
            b -= args.lr_b * gb

            spectral_clip_(W, args.spectral_clip)

    final = {
        "W": W.detach().cpu(),
        "Win": Win.detach().cpu(),
        "b": b.detach().cpu(),
        "h": h.detach().cpu(),
        "h0": h0.detach().cpu(),
        "x": x.detach().cpu(),
        "target": target.detach().cpu(),
        "pred": pred.detach().cpu(),
        "hist": hist,
        "method": method,
        "insertions": insertions,
    }
    return final


# -----------------------------
# plots
# -----------------------------

def impulse_curve(W, max_lag):
    W = W.detach().cpu()
    cur = torch.eye(W.shape[0])
    vals = []
    for _ in range(max_lag + 1):
        vals.append(float(cur.norm()))
        cur = W @ cur
    return np.asarray(vals)


def plot_results(args, results, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1. Training curves.
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
    for name, res in results.items():
        h = res["hist"]
        ax[0].semilogy(h["step"], h["loss"], label=name)
        ax[1].plot(h["step"], h["rmse"], label=name)
        ax[2].plot(h["step"], h["cos"], label=name)
    ax[0].set_title("memory-pro training loss")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("MSE")
    ax[1].set_title("final response RMSE")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("RMSE")
    ax[2].set_title("direction cosine")
    ax[2].set_xlabel("step")
    ax[2].set_ylabel("cosine")
    for a in ax:
        a.grid(True, alpha=0.3)
        a.legend(frameon=False)
    fig.tight_layout()
    path = outdir / "01_training_curves.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    # 2. Prediction scatter.
    fig, ax = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        ax = [ax]
    for a, (name, res) in zip(ax, results.items()):
        target = res["target"]
        pred = res["pred"]
        a.scatter(target[:, 0], pred[:, 0], s=12, alpha=0.65, label="x")
        a.scatter(target[:, 1], pred[:, 1], s=12, alpha=0.65, label="y")
        lim = float(torch.cat([target.flatten(), pred.flatten()]).abs().max()) + 0.1
        a.plot([-lim, lim], [-lim, lim], color="k", lw=1)
        a.set_title(f"{name}: final response")
        a.set_xlabel("target")
        a.set_ylabel("prediction")
        a.grid(True, alpha=0.3)
        a.legend(frameon=False)
    fig.tight_layout()
    path = outdir / "02_final_response_scatter.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    # 3. Example trajectories.
    fig, ax = plt.subplots(2, len(results), figsize=(5 * len(results), 6), sharex=True)
    if len(results) == 1:
        ax = np.asarray(ax).reshape(2, 1)
    t = np.arange(args.T)
    for j, (name, res) in enumerate(results.items()):
        h = res["h"]
        target = res["target"]
        nshow = min(8, h.shape[0])
        for i in range(nshow):
            ax[0, j].plot(t, h[i, :, 0], alpha=0.65)
            ax[1, j].plot(t, h[i, :, 1], alpha=0.65)
        ax[0, j].set_title(f"{name}: h[:,0]")
        ax[1, j].set_title(f"{name}: h[:,1]")
        ax[1, j].set_xlabel("time")
        for a in ax[:, j]:
            a.axvline(args.T - 1, color="k", lw=0.8, ls="--")
            a.grid(True, alpha=0.3)
    fig.tight_layout()
    path = outdir / "03_example_rollouts.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    # 4. Spectrum and memory curve.
    fig, ax = plt.subplots(2, len(results), figsize=(5 * len(results), 7))
    if len(results) == 1:
        ax = np.asarray(ax).reshape(2, 1)
    for j, (name, res) in enumerate(results.items()):
        W = res["W"]
        eig = torch.linalg.eigvals(W).numpy()
        ax[0, j].scatter(eig.real, eig.imag, s=18)
        ax[0, j].add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.5", lw=1))
        ax[0, j].set_aspect("equal", adjustable="box")
        ax[0, j].set_title(f"{name}: W eigenvalues")
        ax[0, j].grid(True, alpha=0.3)

        vals = impulse_curve(W, args.max_lag)
        ax[1, j].plot(vals)
        ax[1, j].set_title(f"{name}: ||W^lag||")
        ax[1, j].set_xlabel("lag")
        ax[1, j].set_ylabel("norm")
        ax[1, j].grid(True, alpha=0.3)
    fig.tight_layout()
    path = outdir / "04_spectrum_memory.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    # 5. Save summary JSON.
    summary = {}
    for name, res in results.items():
        hist = res["hist"]
        summary[name] = {
            "final_loss": hist["loss"][-1],
            "final_rmse": hist["rmse"][-1],
            "final_cos": hist["cos"][-1],
            "final_rho": hist["rho"][-1],
            "insertions": res["insertions"],
        }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    paths.append(outdir / "summary.json")

    torch.save(results, outdir / "training_results.pt")
    paths.append(outdir / "training_results.pt")
    return paths


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="figures_memory_pro_split")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")

    p.add_argument("--n", type=int, default=24)
    p.add_argument("--input-dim", type=int, default=3)
    p.add_argument("--T", type=int, default=12)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--log-every", type=int, default=25)

    p.add_argument("--lambda-w", type=float, default=0.98)
    p.add_argument("--bulk-radius", type=float, default=0.2)
    p.add_argument("--w-scale", type=float, default=0.9)
    p.add_argument("--win-scale", type=float, default=0.4)
    p.add_argument("--cue-scale", type=float, default=1.0)
    p.add_argument("--spectral-clip", type=float, default=1.05)

    p.add_argument("--linear-insertions", type=int, default=1,
                   help="Number of nonlinear insertions when using linear_base.")
    p.add_argument("--residual-insertions", type=int, default=1,
                   help="Number of linear insertions when using residual_base.")
    p.add_argument("--direct-k", type=int, default=3)

    p.add_argument("--lr-W", type=float, default=5e-3)
    p.add_argument("--lr-Win", type=float, default=5e-3)
    p.add_argument("--lr-b", type=float, default=5e-3)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--max-lag", type=int, default=40)

    p.add_argument("--methods", type=str, nargs="+",
                   default=["linear_base", "residual_base"],
                   choices=["linear_base", "residual_base", "direct"])
    args = p.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    if args.quick:
        args.n = min(args.n, 12)
        args.T = min(args.T, 8)
        args.batch = min(args.batch, 64)
        args.steps = min(args.steps, 300)
        args.log_every = max(5, min(args.log_every, 25))
        args.max_lag = min(args.max_lag, 20)
    return args


def main():
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"device={args.device} n={args.n} T={args.T} batch={args.batch} steps={args.steps}")
    print(f"methods={args.methods}")

    t0 = time.perf_counter()
    results = {}
    for method in args.methods:
        if method == "linear_base":
            insertions = args.linear_insertions
        elif method == "residual_base":
            insertions = args.residual_insertions
        else:
            insertions = args.direct_k
        results[method] = train_one(args, method, insertions)

    paths = plot_results(args, results, outdir)

    print(f"elapsed={time.perf_counter() - t0:.2f}s")
    for name, res in results.items():
        h = res["hist"]
        print(
            f"{name:14s} final_loss={h['loss'][-1]:.4e} "
            f"rmse={h['rmse'][-1]:.4f} cos={h['cos'][-1]:.4f}"
        )
    print("saved:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
