#!/usr/bin/env python3
"""Long-run fast split-role experiment.

Goal
----
Show that the Duhamel credit split can push two recurrent matrices into different
roles:

  W1: long-term dependency / memory carrier
  W2: short-term/local correction carrier

The training rule is intentionally asymmetric:

  - W1 is updated using the base long-range credit P1^T e_long.
  - W2 is updated using the Duhamel correction for the short error, mixed with a
    very local credit term.

This script keeps the operator-library style, but uses the fast specialized base
inverse from dbptt_compare_operators_fastbase.py. It avoids per-step expensive
analysis. It logs sparse diagnostics and then runs richer posthoc analysis from
snapshots.

Example:
    python split_role_linear_rnn_longrun.py --steps 12000 --out figures_split_role_12k

Quick test:
    python split_role_linear_rnn_longrun.py --quick --out figures_quick
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
from torch import nn
from tqdm import tqdm

from dbptt_compare_operators_fastbase import (
    DuhamelGreen,
    LinearRNNStateJacobian,
    TruncatedNeumann,
    make_W,
    spectral_clip_,
    step_down,
)


# ----------------------------
# Dynamics / task
# ----------------------------

def rollout(W1, W2, h0, T):
    h = h0
    hs = []
    W = W1 + W2
    for _ in range(T):
        h = h @ W.T
        hs.append(h)
    return torch.stack(hs, dim=1)


def long_target_from_h0(h0, target):
    if target == "quadrant":
        y = torch.sign(h0[:, :2])
        y[y == 0] = 1.0
        return y
    if target == "copy":
        return h0[:, :2]
    raise ValueError(f"Unknown target: {target}")


def task_error(h, h0, short_t, target):
    e = torch.zeros_like(h)
    e_long = torch.zeros_like(h)
    e_short = torch.zeros_like(h)

    long_target = long_target_from_h0(h0, target)
    short_target = h0[:, 2:4]

    long_err = h[:, -1, :2] - long_target
    short_err = h[:, short_t, 2:4] - short_target

    e[:, -1, :2] = long_err / h.shape[0]
    e[:, short_t, 2:4] = short_err / h.shape[0]
    e_long[:, -1, :2] = long_err / h.shape[0]
    e_short[:, short_t, 2:4] = short_err / h.shape[0]

    loss = 0.5 * (long_err.square().mean() + short_err.square().mean())
    long_rmse = long_err.square().mean().sqrt()
    short_rmse = short_err.square().mean().sqrt()
    return loss, e, e_long, e_short, long_rmse, short_rmse


def param_update(credit, h, h0):
    prev = step_down(h, h0)
    return -torch.einsum("btn,btm->nm", credit, prev)


# ----------------------------
# Operator-library credit rule
# ----------------------------

def split_credit_operators(W1, W2, h, w2_insertions, solver_kwargs):
    """Build split credit operators using the project operator library.

    S1 is D_h F for h_{t+1}=W1 h_t, and S is D_h F for h_{t+1}=(W1+W2)h_t.
    The perturbation S1-S corresponds to the W2 state update block. DuhamelGreen
    therefore expands around the W1 Green's function and inserts W2 corrections.
    """
    S1 = LinearRNNStateJacobian(h, W1)
    S = LinearRNNStateJacobian(h, W1 + W2)
    P1 = S1.inverse(solver="neumann", solver_kwargs=solver_kwargs)
    Psplit = DuhamelGreen(P1, S1 - S, w2_insertions + 1)
    Plocal = TruncatedNeumann(S, 2)
    return P1, Psplit, Plocal


def compute_credit_and_updates(args, W1, W2, h, h0, e, e_long, e_short, solver_kwargs):
    P1, Psplit, Plocal = split_credit_operators(W1, W2, h, args.w2_insertions, solver_kwargs)

    if args.separate_errors:
        # W1 gets the final/long error through the long-range W1 Green's operator.
        p1_long = P1.T(e_long)
        p1_short = P1.T(e_short)
        base_credit = p1_long

        # W2 gets the incremental Duhamel correction for the short error, plus a
        # local short credit. This is the actual role-splitting bias.
        split_credit_term = Psplit.T(e_short) - p1_short
        local_credit = Plocal.T(e_short)
    else:
        # Control-ish mode: both errors are pooled before the split.
        base_credit = P1.T(e)
        split_credit_term = Psplit.T(e) - base_credit
        local_credit = Plocal.T(e)

    w2_credit = args.duhamel_mix * split_credit_term + (1.0 - args.duhamel_mix) * local_credit
    dW1 = param_update(base_credit, h, h0)
    dW2 = param_update(w2_credit, h, h0)
    return base_credit, w2_credit, dW1, dW2


# ----------------------------
# Diagnostics
# ----------------------------

def block_norms(W):
    return {
        "long": float(W[:2, :2].detach().norm().cpu()),
        "short": float(W[2:4, 2:4].detach().norm().cpu()),
        "cross": float((W[:2, 2:4].detach().norm() + W[2:4, :2].detach().norm()).cpu() * 0.5),
        "all": float(W.detach().norm().cpu()),
        "rho": float(torch.linalg.eigvals(W.detach()).abs().max().real.cpu()),
    }


def role_metrics(W1, W2):
    b1 = block_norms(W1)
    b2 = block_norms(W2)
    eps = 1e-12
    return {
        "W1_long_frac": b1["long"] / (b1["long"] + b1["short"] + b1["cross"] + eps),
        "W2_short_frac": b2["short"] / (b2["long"] + b2["short"] + b2["cross"] + eps),
        "long_ratio_W1_over_W2": b1["long"] / (b2["long"] + eps),
        "short_ratio_W2_over_W1": b2["short"] / (b1["short"] + eps),
        "rho_W1": b1["rho"],
        "rho_W2": b2["rho"],
        "rho_total": float(torch.linalg.eigvals((W1 + W2).detach()).abs().max().real.cpu()),
    }


def lag_response(W, max_lag):
    cur = torch.eye(W.shape[0], device=W.device)
    vals = []
    long_vals = []
    short_vals = []
    cross_vals = []
    powers = []
    for _ in range(max_lag + 1):
        powers.append(cur.detach().cpu().numpy())
        vals.append(float(cur.norm().detach().cpu()))
        long_vals.append(float(cur[:2, :2].norm().detach().cpu()))
        short_vals.append(float(cur[2:4, 2:4].norm().detach().cpu()))
        cross_vals.append(float(((cur[:2, 2:4].norm() + cur[2:4, :2].norm()) * 0.5).detach().cpu()))
        cur = W @ cur
    vals = np.asarray(vals)
    fft = np.abs(np.fft.rfft(vals - vals.mean()))
    return {
        "all": np.asarray(vals),
        "long": np.asarray(long_vals),
        "short": np.asarray(short_vals),
        "cross": np.asarray(cross_vals),
        "fft": fft,
        "powers": np.stack(powers),
    }


def evaluate(W1, W2, h0, short_t, T, target):
    h = rollout(W1, W2, h0, T)
    long_target = long_target_from_h0(h0, target)
    short_target = h0[:, 2:4]
    long_pred = h[:, -1, :2]
    short_pred = h[:, short_t, 2:4]
    long_acc = ((long_pred > 0) == (long_target > 0)).float().mean().item()
    long_rmse = (long_pred - long_target).square().mean().sqrt().item()
    short_rmse = (short_pred - short_target).square().mean().sqrt().item()
    return h.detach(), long_pred.detach(), long_target.detach(), short_pred.detach(), short_target.detach(), long_acc, long_rmse, short_rmse


# ----------------------------
# Training
# ----------------------------

def init_weights(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if args.role_init:
        W1_init = args.w1_long * torch.eye(args.n, device=args.device)
        W2_init = args.w2_scale * torch.randn(args.n, args.n, device=args.device)
        W2_init[2:4, 2:4] += args.w2_short * torch.eye(2, device=args.device)
    else:
        W1_init = make_W(args.n, args.lambda_w1, args.bulk_radius, rng, args.device)
        W2_init = args.w2_scale * torch.randn(args.n, args.n, device=args.device)
    return W1_init, W2_init


def train(args):
    W1_init, W2_init = init_weights(args)
    W1 = nn.Parameter(W1_init)
    W2 = nn.Parameter(W2_init)
    h0 = torch.randn(args.batch, args.n, device=args.device)

    hist = {
        "step": [],
        "loss": [], "long_rmse": [], "short_rmse": [],
        "W1": [], "W2": [], "role": [],
        "grad_W1": [], "grad_W2": [],
        "credit_long": [], "credit_short": [],
    }
    snapshots = []
    solver_kwargs = {"max_iter": args.T + 1, "tol": 1e-10, "early_stop": False}
    short_t = min(args.short_t, args.T - 1)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    pbar = tqdm(range(args.steps + 1), dynamic_ncols=True)
    for step in pbar:
        h = rollout(W1, W2, h0, args.T)
        loss, e, e_long, e_short, long_rmse, short_rmse = task_error(h, h0, short_t, args.target)
        base_credit, w2_credit, dW1, dW2 = compute_credit_and_updates(
            args, W1, W2, h, h0, e, e_long, e_short, solver_kwargs
        )

        should_log = (step % args.log_every == 0) or (step == args.steps)
        if should_log:
            hist["step"].append(int(step))
            hist["loss"].append(float(loss.detach().cpu()))
            hist["long_rmse"].append(float(long_rmse.detach().cpu()))
            hist["short_rmse"].append(float(short_rmse.detach().cpu()))
            hist["W1"].append(block_norms(W1.detach()))
            hist["W2"].append(block_norms(W2.detach()))
            hist["role"].append(role_metrics(W1.detach(), W2.detach()))
            hist["grad_W1"].append(float(dW1.detach().norm().cpu()))
            hist["grad_W2"].append(float(dW2.detach().norm().cpu()))
            hist["credit_long"].append(float(base_credit.detach().norm().cpu()))
            hist["credit_short"].append(float(w2_credit.detach().norm().cpu()))
            pbar.set_postfix(loss=f"{hist['loss'][-1]:.3e}", role=f"{hist['role'][-1]['W1_long_frac']:.2f}/{hist['role'][-1]['W2_short_frac']:.2f}")

        should_snapshot = args.snapshot_every > 0 and ((step % args.snapshot_every == 0) or (step == args.steps))
        if should_snapshot:
            snap = {
                "step": int(step),
                "W1": W1.detach().cpu().clone(),
                "W2": W2.detach().cpu().clone(),
            }
            snapshots.append(snap)
            if args.save_snapshots:
                torch.save(snap, outdir / f"snapshot_{step:07d}.pt")

        if step == args.steps:
            break

        with torch.no_grad():
            g1 = dW1.norm()
            g2 = dW2.norm()
            if g1 > args.grad_clip:
                dW1 = dW1 * (args.grad_clip / g1)
            if g2 > args.grad_clip:
                dW2 = dW2 * (args.grad_clip / g2)
            W1 += args.lr1 * dW1 - args.weight_decay * W1
            W2 += args.lr2 * dW2 - args.weight_decay * W2
            spectral_clip_(W1, args.spectral_clip)
            spectral_clip_(W2, args.spectral_clip_w2)

    elapsed = time.perf_counter() - t0

    final = {
        "W1": W1.detach().cpu(),
        "W2": W2.detach().cpu(),
        "h0": h0.detach().cpu(),
        "hist": hist,
        "snapshots": snapshots,
        "args": vars(args) | {"device": str(args.device)},
        "elapsed_sec": elapsed,
    }
    torch.save(final, outdir / "training_run.pt")
    return W1.detach(), W2.detach(), hist, h0.detach(), snapshots, elapsed


# ----------------------------
# Plotting
# ----------------------------

def _steps(hist):
    return np.asarray(hist.get("step", np.arange(len(hist["loss"]))))


def plot_training_overview(args, W1, W2, hist, h0, outdir):
    steps = _steps(hist)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))

    ax[0].semilogy(steps, hist["loss"], label="total")
    ax[0].semilogy(steps, hist["long_rmse"], label="long final")
    ax[0].semilogy(steps, hist["short_rmse"], label="short copy")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("loss / RMSE")
    ax[0].legend(frameon=False)
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(steps, [r["W1_long_frac"] for r in hist["role"]], label="W1 long fraction")
    ax[1].plot(steps, [r["W2_short_frac"] for r in hist["role"]], label="W2 short fraction")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("role fraction")
    ax[1].set_ylim(-0.03, 1.03)
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].grid(True, alpha=0.3)

    ax[2].semilogy(steps, hist["grad_W1"], label="grad W1")
    ax[2].semilogy(steps, hist["grad_W2"], label="grad W2")
    ax[2].semilogy(steps, hist["credit_long"], label="W1/base credit")
    ax[2].semilogy(steps, hist["credit_short"], label="W2/short credit")
    ax[2].set_xlabel("step")
    ax[2].set_ylabel("norm")
    ax[2].legend(frameon=False, fontsize=7)
    ax[2].grid(True, alpha=0.3)

    fig.tight_layout()
    path = outdir / "01_training_overview.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_block_norms(args, W1, W2, hist, outdir):
    steps = _steps(hist)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.8))

    for name in ["long", "short", "cross", "all"]:
        ax[0].plot(steps, [x[name] for x in hist["W1"]], label=f"W1 {name}")
    ax[0].set_title("W1 block norms")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("norm")
    ax[0].legend(frameon=False, fontsize=8, ncol=2)
    ax[0].grid(True, alpha=0.3)

    for name in ["long", "short", "cross", "all"]:
        ax[1].plot(steps, [x[name] for x in hist["W2"]], label=f"W2 {name}")
    ax[1].set_title("W2 block norms")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("norm")
    ax[1].legend(frameon=False, fontsize=8, ncol=2)
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = outdir / "02_block_norms.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_role_ratios(args, hist, outdir):
    steps = _steps(hist)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(steps, [r["long_ratio_W1_over_W2"] for r in hist["role"]], label="long block: W1 / W2")
    ax.semilogy(steps, [r["short_ratio_W2_over_W1"] for r in hist["role"]], label="short block: W2 / W1")
    ax.axhline(1.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("role ratio")
    ax.set_title("role specialization ratios")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = outdir / "03_role_ratios.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_lag_memory(args, W1, W2, outdir):
    r1 = lag_response(W1, args.max_lag)
    r2 = lag_response(W2, args.max_lag)
    rt = lag_response(W1 + W2, args.max_lag)
    lag = np.arange(args.max_lag + 1)

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(lag, r1["all"], label="W1")
    ax[0].plot(lag, r2["all"], label="W2")
    ax[0].plot(lag, rt["all"], label="W1+W2", color="k", alpha=0.55)
    ax[0].set_xlabel("lag")
    ax[0].set_ylabel(r"$\|W^\ell\|_F$")
    ax[0].set_title("total impulse memory")
    ax[0].legend(frameon=False)
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(lag, r1["long"], label="W1 long")
    ax[1].plot(lag, r1["short"], label="W1 short")
    ax[1].plot(lag, r2["long"], "--", label="W2 long")
    ax[1].plot(lag, r2["short"], "--", label="W2 short")
    ax[1].set_xlabel("lag")
    ax[1].set_ylabel("block impulse norm")
    ax[1].set_title("which matrix carries which channel?")
    ax[1].legend(frameon=False, fontsize=7, ncol=2)
    ax[1].grid(True, alpha=0.3)

    eps = 1e-12
    ax[2].semilogy(lag, r1["long"] / (r2["long"] + eps), label="long: W1 / W2")
    ax[2].semilogy(lag, r2["short"] / (r1["short"] + eps), label="short: W2 / W1")
    ax[2].axhline(1.0, color="k", lw=0.8, alpha=0.5)
    ax[2].set_xlabel("lag")
    ax[2].set_ylabel("impulse ratio")
    ax[2].set_title("lag-dependent role ratio")
    ax[2].legend(frameon=False, fontsize=8)
    ax[2].grid(True, alpha=0.3)

    fig.tight_layout()
    path = outdir / "04_lag_memory_roles.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_weights_and_spectrum(args, W1, W2, outdir):
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.5))
    mats = [(W1, "W1"), (W2, "W2"), (W1 + W2, "W1 + W2")]
    vmax = max(float(W1.abs().max().cpu()), float(W2.abs().max().cpu()), float((W1 + W2).abs().max().cpu()), 1e-6)
    for j, (W, title) in enumerate(mats):
        im = ax[0, j].imshow(W.detach().cpu(), cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax[0, j].set_title(title)
        ax[0, j].axhline(1.5, color="k", lw=0.8)
        ax[0, j].axvline(1.5, color="k", lw=0.8)
        eig = torch.linalg.eigvals(W.detach()).cpu().numpy()
        ax[1, j].scatter(eig.real, eig.imag, s=18)
        ax[1, j].add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.5", lw=1))
        ax[1, j].set_aspect("equal", adjustable="box")
        ax[1, j].set_title(f"{title} eigenvalues")
        ax[1, j].grid(True, alpha=0.3)
    fig.colorbar(im, ax=ax[0].ravel().tolist(), shrink=0.75)
    fig.tight_layout()
    path = outdir / "05_weights_spectrum.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_predictions(args, W1, W2, h0, outdir):
    short_t = min(args.short_t, args.T - 1)
    h, long_pred, long_target, short_pred, short_target, long_acc, long_rmse, short_rmse = evaluate(
        W1, W2, h0, short_t, args.T, args.target
    )
    nshow = min(8, h0.shape[0])
    t = np.arange(args.T)

    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    for i in range(nshow):
        ax[0, 0].plot(t, h[i, :, 0].detach().cpu(), alpha=0.65)
        ax[0, 1].plot(t, h[i, :, 2].detach().cpu(), alpha=0.65)
    ax[0, 0].axhline(1, color="k", lw=0.8, ls="--")
    ax[0, 0].axhline(-1, color="k", lw=0.8, ls="--")
    ax[0, 0].set_title("example long-channel rollouts")
    ax[0, 1].axvline(short_t, color="k", lw=0.8, ls="--")
    ax[0, 1].set_title("example short-channel rollouts")

    ax[1, 0].scatter(long_target[:, 0].cpu(), long_pred[:, 0].cpu(), s=14, alpha=0.7)
    ax[1, 0].scatter(long_target[:, 1].cpu(), long_pred[:, 1].cpu(), s=14, alpha=0.7)
    ax[1, 0].plot([-1.2, 1.2], [-1.2, 1.2], color="k", lw=0.8)
    ax[1, 0].set_title(f"final {args.target}, rmse={long_rmse:.2f}, sign={long_acc:.2f}")
    ax[1, 0].set_xlabel("target")
    ax[1, 0].set_ylabel("prediction")

    ax[1, 1].scatter(short_target[:, 0].cpu(), short_pred[:, 0].cpu(), s=14, alpha=0.7)
    ax[1, 1].scatter(short_target[:, 1].cpu(), short_pred[:, 1].cpu(), s=14, alpha=0.7)
    lim = float(torch.cat([short_target.flatten(), short_pred.flatten()]).abs().max().cpu()) + 0.1
    ax[1, 1].plot([-lim, lim], [-lim, lim], color="k", lw=0.8)
    ax[1, 1].set_title(f"short copy at t={short_t}, rmse={short_rmse:.2f}")
    ax[1, 1].set_xlabel("target")
    ax[1, 1].set_ylabel("prediction")

    for a in ax.ravel():
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    path = outdir / "06_predictions_examples.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_snapshot_evolution(args, snapshots, outdir):
    if len(snapshots) < 2:
        return None
    steps = np.asarray([s["step"] for s in snapshots])
    lags = sorted(set([1, min(args.short_t, args.T - 1), max(1, args.T // 2), args.T]))
    W1_lag = {lag: [] for lag in lags}
    W2_lag = {lag: [] for lag in lags}
    total_lag = {lag: [] for lag in lags}
    role_w1_long = []
    role_w2_short = []

    for snap in snapshots:
        W1 = snap["W1"].to(args.device)
        W2 = snap["W2"].to(args.device)
        role = role_metrics(W1, W2)
        role_w1_long.append(role["W1_long_frac"])
        role_w2_short.append(role["W2_short_frac"])
        cur1 = torch.eye(W1.shape[0], device=args.device)
        cur2 = torch.eye(W1.shape[0], device=args.device)
        curt = torch.eye(W1.shape[0], device=args.device)
        for ell in range(max(lags) + 1):
            if ell in lags:
                W1_lag[ell].append(float(cur1.norm().cpu()))
                W2_lag[ell].append(float(cur2.norm().cpu()))
                total_lag[ell].append(float(curt.norm().cpu()))
            cur1 = W1 @ cur1
            cur2 = W2 @ cur2
            curt = (W1 + W2) @ curt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(steps, role_w1_long, label="W1 long fraction")
    ax[0].plot(steps, role_w2_short, label="W2 short fraction")
    ax[0].set_ylim(-0.03, 1.03)
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("role fraction")
    ax[0].set_title("snapshot role evolution")
    ax[0].legend(frameon=False)
    ax[0].grid(True, alpha=0.3)

    for ell in lags:
        ax[1].plot(steps, W1_lag[ell], label=f"W1 lag {ell}")
        ax[1].plot(steps, W2_lag[ell], "--", label=f"W2 lag {ell}")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel(r"$\|W^\ell\|_F$")
    ax[1].set_title("memory by lag over training")
    ax[1].legend(frameon=False, fontsize=7, ncol=2)
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = outdir / "07_snapshot_memory_evolution.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_summary(args, W1, W2, hist, snapshots, elapsed, outdir):
    final_role = role_metrics(W1, W2)
    final_W1 = block_norms(W1)
    final_W2 = block_norms(W2)
    summary = {
        "elapsed_sec": elapsed,
        "steps": args.steps,
        "logs": len(hist["loss"]),
        "snapshots": len(snapshots),
        "final_loss": hist["loss"][-1],
        "final_long_rmse": hist["long_rmse"][-1],
        "final_short_rmse": hist["short_rmse"][-1],
        "final_role": final_role,
        "final_W1_block_norms": final_W1,
        "final_W2_block_norms": final_W2,
        "interpretation": {
            "W1_long_fraction": "larger means W1 is concentrated in the final/long channel block",
            "W2_short_fraction": "larger means W2 is concentrated in the short/copy channel block",
            "long_ratio_W1_over_W2": "larger than 1 means W1 dominates the long block",
            "short_ratio_W2_over_W1": "larger than 1 means W2 dominates the short block",
        },
    }
    path = outdir / "summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path


def plot_all(args, W1, W2, hist, h0, snapshots, elapsed):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    paths.append(plot_training_overview(args, W1, W2, hist, h0, outdir))
    paths.append(plot_block_norms(args, W1, W2, hist, outdir))
    paths.append(plot_role_ratios(args, hist, outdir))
    paths.append(plot_lag_memory(args, W1, W2, outdir))
    paths.append(plot_weights_and_spectrum(args, W1, W2, outdir))
    paths.append(plot_predictions(args, W1, W2, h0, outdir))
    snap_path = plot_snapshot_evolution(args, snapshots, outdir)
    if snap_path is not None:
        paths.append(snap_path)
    paths.append(write_summary(args, W1, W2, hist, snapshots, elapsed, outdir))
    return paths


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="figures_split_role_longrun")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")

    p.add_argument("--n", type=int, default=8)
    p.add_argument("--T", type=int, default=24)
    p.add_argument("--short-t", type=int, default=2)
    p.add_argument("--target", type=str, choices=["copy", "quadrant"], default="copy")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--steps", type=int, default=12000)

    p.add_argument("--lr1", type=float, default=4e-2)
    p.add_argument("--lr2", type=float, default=5e-2)
    p.add_argument("--lambda-w1", type=float, default=0.96)
    p.add_argument("--bulk-radius", type=float, default=0.25)
    p.add_argument("--w2-scale", type=float, default=0.03)
    p.add_argument("--role-init", action="store_true")
    p.add_argument("--w1-long", type=float, default=0.98)
    p.add_argument("--w2-short", type=float, default=0.25)

    p.add_argument("--w2-insertions", type=int, default=1)
    p.add_argument("--duhamel-mix", type=float, default=0.7)
    p.add_argument("--shared-errors", action="store_true", help="Use pooled errors instead of separate long/short credit channels.")

    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--spectral-clip", type=float, default=1.05)
    p.add_argument("--spectral-clip-w2", type=float, default=0.35)

    p.add_argument("--max-lag", type=int, default=48)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--snapshot-every", type=int, default=500)
    p.add_argument("--save-snapshots", action="store_true")
    p.add_argument("--no-plots", action="store_true")

    args = p.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.quick:
        args.n = min(args.n, 6)
        args.T = min(args.T, 16)
        args.batch = min(args.batch, 64)
        args.steps = min(args.steps, 300)
        args.max_lag = min(args.max_lag, 24)
        args.log_every = min(args.log_every, 10)
        args.snapshot_every = min(args.snapshot_every, 50)
    args.separate_errors = not args.shared_errors
    return args


def main():
    args = parse_args()
    print(f"device={args.device} steps={args.steps} T={args.T} batch={args.batch} out={args.out}", flush=True)
    print(f"rule: separate_errors={args.separate_errors} w2_insertions={args.w2_insertions} duhamel_mix={args.duhamel_mix}", flush=True)
    W1, W2, hist, h0, snapshots, elapsed = train(args)
    paths = [] if args.no_plots else plot_all(args, W1, W2, hist, h0, snapshots, elapsed)

    print(f"final_loss={hist['loss'][-1]:.4e} long_rmse={hist['long_rmse'][-1]:.4e} short_rmse={hist['short_rmse'][-1]:.4e}")
    print("final W1 block norms:", hist["W1"][-1])
    print("final W2 block norms:", hist["W2"][-1])
    print("final role metrics:", hist["role"][-1])
    print(f"elapsed={elapsed:.2f}s")
    if paths:
        print("saved:")
        for path in paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
