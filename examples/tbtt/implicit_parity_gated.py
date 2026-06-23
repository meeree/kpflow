#!/usr/bin/env python3
"""Temporal parity: exact BPTT, direct TBPTT, and one-insertion Duhamel."""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm.auto import trange

from kpflow.op_common import LinearOperator


def parity_batch(batch, T, device, event_probability):
    events = torch.rand(batch, T, device=device) < event_probability
    signs = (2 * torch.randint(0, 2, (batch, T), device=device) - 1).float()
    x = (events * signs).unsqueeze(-1)
    target = torch.where(events, signs, torch.ones_like(signs)).prod(dim=1, keepdim=True)
    return x, target


class GatedParityRNN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_f = nn.Linear(1, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_c = nn.Linear(1, hidden_size)
        self.U_c = nn.Linear(hidden_size, hidden_size, bias=False)
        self.readout = nn.Linear(hidden_size, 1)
        nn.init.constant_(self.W_f.bias, 1.5)

    def step(self, h, x):
        forget = torch.sigmoid(self.W_f(x) + self.U_f(h))
        candidate = torch.tanh(self.W_c(x) + self.U_c(h))
        return forget * h + (1.0 - forget) * candidate, forget

    def rollout(self, x):
        h = torch.zeros(x.shape[0], self.hidden_size, dtype=x.dtype, device=x.device)
        hs, gates = [], []
        for t in range(x.shape[1]):
            h, gate = self.step(h, x[:, t])
            hs.append(h)
            gates.append(gate)
        return torch.stack(hs, 1), torch.stack(gates, 1)

    def forward(self, x):
        h, gates = self.rollout(x)
        return torch.tanh(self.readout(h[:, -1])), h, gates


def step_down(h):
    return torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], 1)


def shift_up(h):
    return torch.cat([h[:, 1:], torch.zeros_like(h[:, :1])], 1)


class IdentityGreen(LinearOperator):
    """Exact Green's function for h_t - h_{t-1}=b_t."""

    def __init__(self, shape, device):
        super().__init__(shape, shape, dev=device)

    def _matvec(self, b):
        return torch.cumsum(b, dim=1)

    def _rmatvec(self, b):
        return torch.flip(torch.cumsum(torch.flip(b, [1]), dim=1), [1])


class GatedStateJacobian(LinearOperator):
    """D_h F for F_t=h_t-g(h_{t-1}, x_t), with hardcoded JVP/VJP."""

    def __init__(self, h, x, model):
        self.h_prev = step_down(h)
        self.x = x
        self.model = model
        with torch.no_grad():
            self.forget = torch.sigmoid(model.W_f(x) + model.U_f(self.h_prev))
            self.candidate = torch.tanh(model.W_c(x) + model.U_c(self.h_prev))
        super().__init__(h.shape, h.shape, dev=h.device)

    def jvp(self, dh):
        df = self.forget * (1.0 - self.forget) * self.model.U_f(dh)
        dc = (1.0 - self.candidate.square()) * self.model.U_c(dh)
        return self.forget * dh + (self.h_prev - self.candidate) * df + (1.0 - self.forget) * dc

    def vjp(self, q):
        df = q * (self.h_prev - self.candidate) * self.forget * (1.0 - self.forget)
        dc = q * (1.0 - self.forget) * (1.0 - self.candidate.square())
        return self.forget * q + df @ self.model.U_f.weight + dc @ self.model.U_c.weight

    def _matvec(self, dh):
        return dh - self.jvp(step_down(dh))

    def _rmatvec(self, q):
        return q - self.vjp(shift_up(q))


class DirectGreen(LinearOperator):
    """Direct Neumann/TBPTT inverse of a causal state Jacobian."""

    def __init__(self, S, terms):
        super().__init__(S.shape_out, S.shape_in, dev=S.dev)
        self.S = S
        self.terms = int(terms)

    def _matvec(self, b):
        term = out = b
        for _ in range(1, self.terms):
            term = term - self.S(term)
            out = out + term
        return out

    def _rmatvec(self, b):
        term = out = b
        ST = self.S.T
        for _ in range(1, self.terms):
            term = term - ST(term)
            out = out + term
        return out


class GatedNonlinearPiece(LinearOperator):
    """S_identity - S_full = shift((Dg - I) dh)."""

    def __init__(self, state_jacobian):
        self.J = state_jacobian
        super().__init__(self.J.shape_in, self.J.shape_out, dev=self.J.dev)

    def _matvec(self, dh):
        prev = step_down(dh)
        return self.J.jvp(prev) - prev

    def _rmatvec(self, q):
        future = shift_up(q)
        return self.J.vjp(future) - future


class DuhamelGreen(LinearOperator):
    """sum_{j=0}^{k-1} (P_base R)^j P_base."""

    def __init__(self, P_base, perturbation, terms):
        super().__init__(P_base.shape_in, P_base.shape_out, dev=P_base.dev)
        self.P_base = P_base
        self.perturbation = perturbation
        self.terms = int(terms)

    def _matvec(self, b):
        term = out = self.P_base(b)
        for _ in range(1, self.terms):
            term = self.P_base(self.perturbation(term))
            out = out + term
        return out

    def _rmatvec(self, b):
        term = out = self.P_base.T(b)
        for _ in range(1, self.terms):
            term = self.P_base.T(self.perturbation.T(term))
            out = out + term
        return out


def output_error(model, h, target):
    pred = torch.tanh(model.readout(h[:, -1]))
    loss = (pred - target).square().mean()
    dz = 2.0 * (pred - target) * (1.0 - pred.square()) / target.shape[0]
    error = torch.zeros_like(h)
    error[:, -1] = dz @ model.readout.weight
    accuracy = (pred.sign() == target).float().mean()
    return loss, error, accuracy


def local_surrogate_gradient(model, x, h, credit, target):
    """Parameter gradient sum_t credit_t d g_t / d theta, without BPTT."""
    model.zero_grad(set_to_none=True)
    prev = torch.zeros_like(h[:, 0])
    surrogate = 0.0
    for t in range(x.shape[1]):
        if t:
            prev = h[:, t - 1].detach()
        next_h, _ = model.step(prev, x[:, t])
        surrogate = surrogate + (next_h * credit[:, t].detach()).sum()
    surrogate.backward()
    readout_loss = (torch.tanh(model.readout(h[:, -1].detach())) - target).square().mean()
    readout_loss.backward()


def operator_step(model, optimizer, x, target, method, args):
    with torch.no_grad():
        h, _ = model.rollout(x)
        loss, error, accuracy = output_error(model, h, target)
        S = GatedStateJacobian(h, x, model)
        if method == "tbtt":
            green = DirectGreen(S, args.tbtt_terms)
        else:
            base = IdentityGreen(h.shape, h.device)
            green = DuhamelGreen(base, GatedNonlinearPiece(S), args.duhamel_terms)
        credit = green.T(error)
    local_surrogate_gradient(model, x, h, credit, target)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    return float(loss), float(accuracy)


def bptt_step(model, optimizer, x, target, args):
    optimizer.zero_grad(set_to_none=True)
    pred, _, _ = model(x)
    loss = (pred - target).square().mean()
    accuracy = (pred.sign() == target).float().mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    return float(loss.detach()), float(accuracy.detach())


def snapshot(path, states, history, step):
    torch.save(
        {
            "step": step,
            "models": {name: state["model"].state_dict() for name, state in states.items()},
            "optimizers": {name: state["optimizer"].state_dict() for name, state in states.items()},
            "history": history,
        },
        path,
    )


def train(args):
    torch.manual_seed(args.seed)
    base = GatedParityRNN(args.hidden_size).to(args.device)
    states = {}
    for name in ("bptt", "tbtt", "duhamel"):
        model = GatedParityRNN(args.hidden_size).to(args.device)
        model.load_state_dict(base.state_dict())
        states[name] = {"model": model, "optimizer": torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)}
    history = {name: {"step": [], "loss": [], "accuracy": []} for name in states}
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    progress = trange(args.steps + 1, desc="parity compare", dynamic_ncols=True, file=sys.stdout, mininterval=1.0)
    for step in progress:
        x, target = parity_batch(args.batch, args.T, args.device, args.event_probability)
        metrics = {}
        for name, state in states.items():
            if name == "bptt":
                metrics[name] = bptt_step(state["model"], state["optimizer"], x, target, args)
            else:
                metrics[name] = operator_step(state["model"], state["optimizer"], x, target, name, args)

        if step % args.log_every == 0 or step == args.steps:
            for name, (loss, accuracy) in metrics.items():
                history[name]["step"].append(step)
                history[name]["loss"].append(loss)
                history[name]["accuracy"].append(accuracy)
            progress.set_postfix(
                bptt=f"{metrics['bptt'][0]:.2e}",
                tbtt=f"{metrics['tbtt'][0]:.2e}",
                duh=f"{metrics['duhamel'][0]:.2e}",
            )
            print(
                f"step={step:05d} "
                + " ".join(f"{name}:loss={loss:.3e},acc={acc:.3f}" for name, (loss, acc) in metrics.items()),
                flush=True,
            )

        if step % args.snapshot_every == 0 or step == args.steps:
            snapshot(args.out / f"snapshot_{step:07d}.pt", states, history, step)
        if step == args.steps:
            break

    torch.save({"models": {name: s["model"].state_dict() for name, s in states.items()}, "history": history}, args.out / "training_history.pt")
    return states, history, time.perf_counter() - t0


def plot_results(states, history, args):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for name, values in history.items():
        ax[0].semilogy(values["step"], values["loss"], label=name)
        ax[1].plot(values["step"], values["accuracy"], label=name)
    ax[0].set_title("temporal parity loss")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("MSE")
    ax[1].set_title("parity accuracy")
    ax[1].set_xlabel("step")
    ax[1].set_ylim(0, 1.02)
    for a in ax:
        a.grid(True, alpha=0.3)
        a.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out / "01_training_comparison.png", dpi=220)
    plt.close(fig)

    x, target = parity_batch(args.eval_batch, args.T, args.device, args.event_probability)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    for a, (name, state) in zip(ax, states.items()):
        with torch.no_grad():
            pred, _, gates = state["model"](x)
            acc = (pred.sign() == target).float().mean()
        a.scatter(target.cpu(), pred.cpu(), s=8, alpha=0.45)
        a.plot([-1.1, 1.1], [-1.1, 1.1], "k--", lw=1)
        a.set_title(f"{name}, acc={float(acc):.3f}")
        a.set_xlabel("target")
        a.set_ylabel("prediction")
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "02_prediction_comparison.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for name, state in states.items():
        with torch.no_grad():
            _, _, gates = state["model"](x[:16])
        ax.plot(gates.mean((0, 2)).cpu(), label=name)
    ax.set_title("mean forget gate over time")
    ax.set_xlabel("time")
    ax.set_ylabel("forget gate")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out / "03_gate_comparison.png", dpi=220)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("figures_implicit_parity_compare"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--T", type=int, default=40)
    p.add_argument("--hidden-size", type=int, default=96)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--event-probability", type=float, default=0.12)
    p.add_argument("--tbtt-terms", type=int, default=2)
    p.add_argument("--duhamel-terms", type=int, default=2, help="Base plus one nonlinear insertion.")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--snapshot-every", type=int, default=50)
    p.add_argument("--eval-batch", type=int, default=512)
    args = p.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.quick:
        args.T = min(args.T, 24)
        args.hidden_size = min(args.hidden_size, 32)
        args.batch = min(args.batch, 128)
        args.steps = min(args.steps, 200)
        args.eval_batch = min(args.eval_batch, 256)
    return args


def main():
    args = parse_args()
    print(f"device={args.device} T={args.T} hidden={args.hidden_size} batch={args.batch} steps={args.steps}")
    states, history, seconds = train(args)
    plot_results(states, history, args)
    print(f"training_seconds={seconds:.2f}")
    for name, values in history.items():
        print(f"{name}: final_loss={values['loss'][-1]:.4e} final_accuracy={values['accuracy'][-1]:.3f}")
    print(f"saved snapshots and comparison figures to {args.out}")


if __name__ == "__main__":
    main()
