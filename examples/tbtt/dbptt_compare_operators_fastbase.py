import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from kpflow.op_common import LinearOperator
from kpflow.pytree_shape import ShapeSpec

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


def step_down(h, h0=None):
    first = torch.zeros_like(h[:, :1]) if h0 is None else h0.unsqueeze(1)
    return torch.cat([first, h[:, :-1]], dim=1)


def shift_up(x):
    return torch.cat([x[:, 1:], torch.zeros_like(x[:, :1])], dim=1)


def cosine(a, b):
    a = a.detach().reshape(-1)
    b = b.detach().reshape(-1)
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def rel_fro_err(A, B):
    return float((A - B).norm() / (B.norm() + 1e-12))


def orthogonal_matrix(n, rng, device):
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return torch.tensor(q * signs, dtype=torch.float32, device=device)


def make_W(n, lambda_dom, bulk_radius, rng, device):
    q = orthogonal_matrix(n, rng, device)
    bulk = torch.tensor(rng.uniform(-bulk_radius, bulk_radius, size=n - 1), dtype=torch.float32, device=device)
    eigs = torch.cat([torch.tensor([lambda_dom], dtype=torch.float32, device=device), bulk])
    return q @ torch.diag(eigs) @ q.T


def spectral_clip_(W, max_abs):
    with torch.no_grad():
        rho = torch.linalg.eigvals(W).abs().max().real
        if rho > max_abs:
            W.mul_(max_abs / rho)


class LinearRNNStateJacobian(LinearOperator):
    def __init__(self, h, W):
        self.W = W
        super().__init__(h.shape, h.shape, dev=h.device)

    def _matvec(self, dh):
        return dh - step_down(dh) @ self.W.T

    def _rmatvec(self, w):
        return w - shift_up(w) @ self.W

    def inverse(self, solver="neumann", solver_kwargs=None, **kwargs):
        """Exact finite-horizon inverse for the linear RNN constraint.

        For S = I - shift_down(·) W^T, the inverse can be applied by a
        triangular recurrence instead of the generic iterative inverse wrapper.
        This keeps the same LinearOperator interface, but makes P.T(e) fast
        in the training loop. The finite horizon makes this exact regardless
        of whether W has eigenvalues above one.
        """
        return LinearRNNGreen(self.shape_in, self.W, self.dev)

    def __str__(self):
        return "D_hF[linear]"


class LinearRNNGreen(LinearOperator):
    """Exact Green's operator for LinearRNNStateJacobian.

    If S(h) = h - step_down(h) @ W.T, this represents P=S^{-1}.

    P b is computed by the forward recurrence
        y_t = b_t + y_{t-1} W.T.

    P.T b is computed by the backward recurrence
        y_t = b_t + y_{t+1} W.
    """

    def __init__(self, shape, W, device):
        self.W = W
        super().__init__(shape, shape, dev=device)

    def _matvec(self, b):
        B, T, N = b.shape
        ys = []
        prev = torch.zeros((B, N), dtype=b.dtype, device=b.device)
        WT = self.W.T
        for t in range(T):
            prev = b[:, t] + prev @ WT
            ys.append(prev)
        return torch.stack(ys, dim=1)

    def _rmatvec(self, b):
        B, T, N = b.shape
        ys = [None] * T
        nxt = torch.zeros((B, N), dtype=b.dtype, device=b.device)
        W = self.W
        for t in range(T - 1, -1, -1):
            nxt = b[:, t] + nxt @ W
            ys[t] = nxt
        return torch.stack(ys, dim=1)

    def __str__(self):
        return "Green[D_hF linear exact]"


class LinearRNNParamJacobian(LinearOperator):
    def __init__(self, h, h0, W):
        self.prev = step_down(h, h0)
        self.theta = {"W": W}
        super().__init__(ShapeSpec.from_tree(self.theta), h.shape, dev=h.device)

    def _matvec(self, dtheta):
        return -(self.prev @ dtheta["W"].T)

    def _rmatvec(self, w):
        return {"W": -torch.einsum("btn,btm->nm", w, self.prev)}

    def __str__(self):
        return "D_WF[linear]"


class CubicStateJacobian(LinearOperator):
    def __init__(self, h, h0, alpha):
        self.coeff = 3.0 * alpha * step_down(h, h0) ** 2
        super().__init__(h.shape, h.shape, dev=h.device)

    def _matvec(self, dh):
        return self.coeff * step_down(dh)

    def _rmatvec(self, w):
        return shift_up(self.coeff * w)

    def __str__(self):
        return "D_hF[cubic]"


class TruncatedNeumann(LinearOperator):
    def __init__(self, S, truncation):
        super().__init__(S.shape_out, S.shape_in, dev=S.dev)
        self.S = S
        self.truncation = int(truncation)

    def _matvec(self, b):
        out = b
        term = b
        for _ in range(1, self.truncation):
            term = term - self.S(term)
            out = out + term
        return out

    def _rmatvec(self, b):
        out = b
        term = b
        ST = self.S.T
        for _ in range(1, self.truncation):
            term = term - ST(term)
            out = out + term
        return out

    def __str__(self):
        return f"Neumann({self.S}, k={self.truncation})"


class DuhamelGreen(LinearOperator):
    def __init__(self, P_base, perturbation, truncation):
        super().__init__(P_base.shape_in, P_base.shape_out, dev=P_base.dev)
        self.P_base = P_base
        self.perturbation = perturbation
        self.truncation = int(truncation)

    def _matvec(self, b):
        term = self.P_base(b)
        out = term
        for _ in range(1, self.truncation):
            term = self.P_base(self.perturbation(term))
            out = out + term
        return out

    def _rmatvec(self, b):
        term = self.P_base.T(b)
        out = term
        for _ in range(1, self.truncation):
            term = self.P_base.T(self.perturbation.T(term))
            out = out + term
        return out

    def __str__(self):
        return f"Duhamel({self.P_base}, k={self.truncation})"


class LinearRNNConstraint:
    def __init__(self, h, h0, W):
        self.h = h
        self.h0 = h0
        self.W = W
        self.example_tuple_inp = (h, {"W": W}, {"h0": h0})

    def __call__(self, primals=None):
        h, theta, x = self.example_tuple_inp if primals is None else primals
        return h - step_down(h, x["h0"]) @ theta["W"].T

    def state_jac(self, primals=None):
        h, theta, _ = self.example_tuple_inp if primals is None else primals
        return LinearRNNStateJacobian(h, theta["W"])

    def param_jac(self, primals=None):
        h, theta, x = self.example_tuple_inp if primals is None else primals
        return LinearRNNParamJacobian(h, x["h0"], theta["W"])

    def greens(self, solver="neumann", truncation=None, **solver_kwargs):
        if truncation is not None:
            return TruncatedNeumann(self.state_jac(), truncation)
        return self.state_jac().inverse(solver=solver, solver_kwargs=solver_kwargs)


class CubicConstraint:
    def __init__(self, h, h0, alpha):
        self.h = h
        self.h0 = h0
        self.alpha = torch.as_tensor(alpha, dtype=h.dtype, device=h.device)
        self.example_tuple_inp = (h, {}, {"h0": h0})

    def __call__(self, primals=None):
        h, _, x = self.example_tuple_inp if primals is None else primals
        return self.alpha * step_down(h, x["h0"]) ** 3

    def state_jac(self, primals=None):
        h, _, x = self.example_tuple_inp if primals is None else primals
        return CubicStateJacobian(h, x["h0"], self.alpha)


class SplitPitchforkConstraint:
    def __init__(self, h, h0, W, alpha):
        self.linear = LinearRNNConstraint(h, h0, W)
        self.cubic = CubicConstraint(h, h0, alpha)
        self.example_tuple_inp = self.linear.example_tuple_inp

    def __call__(self, primals=None):
        h, theta, x = self.example_tuple_inp if primals is None else primals
        return self.linear((h, theta, x)) + self.cubic((h, {}, x))

    def state_jac(self, primals=None):
        h, theta, x = self.example_tuple_inp if primals is None else primals
        return self.linear.state_jac((h, theta, x)) + self.cubic.state_jac((h, {}, x))

    def param_jac(self, primals=None):
        return self.linear.param_jac(self.example_tuple_inp if primals is None else primals)

    def greens(self, method="full", truncation=None, solver="neumann", **solver_kwargs):
        S = self.state_jac()
        if method == "full":
            if truncation is not None:
                return TruncatedNeumann(S, truncation)
            return S.inverse(solver=solver, solver_kwargs=solver_kwargs)
        if method == "direct":
            return TruncatedNeumann(S, truncation)

        if method == "dw":
            base = self.linear.state_jac()
        elif method == "dj":
            base = self.cubic.state_jac() + IdentityState(self.linear.h.shape, self.linear.h.device)
        else:
            raise ValueError(f"Unknown Green's method: {method}")

        if truncation is None:
            raise ValueError(f"{method} needs truncation=<order>.")
        P_base = base.inverse(solver=solver, solver_kwargs=solver_kwargs)
        perturbation = base - S
        return DuhamelGreen(P_base, perturbation, truncation)


class IdentityState(LinearOperator):
    def __init__(self, shape, device):
        super().__init__(shape, shape, dev=device, self_adjoint=True)

    def _matvec(self, q):
        return q


class PitchforkRNN(nn.Module):
    def __init__(self, W, alpha):
        super().__init__()
        self.W = nn.Parameter(W.clone())
        self.alpha = torch.as_tensor(alpha, dtype=W.dtype, device=W.device)

    def forward(self, h):
        return h @ self.W.T - self.alpha * h ** 3

    def rollout(self, h0, T):
        h = h0
        hidden = []
        for _ in range(T):
            h = self(h)
            hidden.append(h)
        return torch.stack(hidden, dim=1)

    def to_implicit(self, h, h0):
        return SplitPitchforkConstraint(h, h0, self.W, self.alpha)


def final_error_like(h, c, err=1.0):
    e = torch.zeros_like(h)
    e[:, -1] = err * c
    return e


def simulate(W, alpha, h0, T):
    return PitchforkRNN(W, alpha).rollout(h0, T).detach()


def operator_suite(W, h, h0, alpha, truncation, solver_kwargs):
    constraint = SplitPitchforkConstraint(h, h0, W, alpha)
    return {
        "full": constraint.greens(method="full", **solver_kwargs),
        "direct": constraint.greens(method="direct", truncation=truncation),
        "dw": constraint.greens(method="dw", truncation=truncation, **solver_kwargs),
        "dj": constraint.greens(method="dj", truncation=truncation, **solver_kwargs),
        "constraint": constraint,
    }


def compute_all(args, T=None, max_k=None, seed_offset=0):
    T = args.T if T is None else int(T)
    max_k = args.max_k if max_k is None else int(max_k)
    rng = np.random.default_rng(args.seed + seed_offset)
    W = make_W(args.n, args.lambda_dom, args.bulk_radius, rng, args.device)
    h0 = args.h0_scale * torch.tensor(rng.normal(size=(1, args.n)), dtype=torch.float32, device=args.device)
    h = simulate(W, args.alpha, h0, T)
    solver_kwargs = {"max_iter": T + 1, "tol": 1e-10, "early_stop": False}
    exact = operator_suite(W, h, h0, args.alpha, max_k, solver_kwargs)["full"]
    suites = [operator_suite(W, h, h0, args.alpha, k, solver_kwargs) for k in range(1, max_k + 1)]
    return W, h, h0, exact, suites


def plot_operator_error_vs_order(args, outdir):
    t0 = time.perf_counter()
    _, _, _, exact, suites = compute_all(args)
    P = exact.full_matrix()
    ks = np.arange(1, args.max_k + 1)
    timing = {}
    err = {"direct": [], "dw": [], "dj": []}
    for method in err:
        start = time.perf_counter()
        for suite in suites:
            err[method].append(rel_fro_err(suite[method].full_matrix(), P))
        timing[f"full_matrix_{method}"] = time.perf_counter() - start

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.semilogy(ks, err["direct"], marker="o", label="Direct / TBPTT")
    ax.semilogy(ks, err["dw"], marker="o", label="DW-BP: exact W")
    ax.semilogy(ks, err["dj"], marker="o", label="DJ-BP: exact J")
    ax.set_xlabel("Truncation / Duhamel order k")
    ax.set_ylabel("Relative operator error")
    ax.set_title("Operator error with hardcoded split operators")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    path = outdir / "operator_error_vs_order.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    timing["plot_operator_error_vs_order"] = time.perf_counter() - t0
    return path, timing


def plot_truncation_sweep_credit(args, outdir):
    t0 = time.perf_counter()
    rng = np.random.default_rng(args.seed + 100)
    _, h, _, exact, suites = compute_all(args, seed_offset=10)
    c = torch.tensor(rng.normal(size=args.n), dtype=torch.float32, device=args.device)
    c = c / c.norm()
    e = final_error_like(h, c)
    full_credit = exact.T(e)
    ks = np.arange(1, args.max_k + 1)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for method, label in [("direct", "Direct / TBPTT"), ("dw", "DW-BP: exact W"), ("dj", "DJ-BP: exact J")]:
        ax.plot(ks, [cosine(suite[method].T(e), full_credit) for suite in suites], marker="o", label=label)
    ax.set_xlabel("Truncation / Duhamel order k")
    ax.set_ylabel("Credit alignment")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    path = outdir / "truncation_sweep_credit_alignment.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path, {"plot_truncation_sweep_credit": time.perf_counter() - t0}


def plot_credit_alignment_vs_delay(args, outdir):
    t0 = time.perf_counter()
    rng = np.random.default_rng(args.seed + 200)
    values = {"direct": [], "dw": [], "dj": []}
    for T in tqdm(args.delay_grid, desc="delay sweep", leave=False):
        _, h, _, exact, suites = compute_all(args, T=T, max_k=args.budget_k, seed_offset=int(T))
        c = torch.tensor(rng.normal(size=args.n), dtype=torch.float32, device=args.device)
        c = c / c.norm()
        e = final_error_like(h, c)
        full_credit = exact.T(e)
        suite = suites[args.budget_k - 1]
        for method in values:
            values[method].append(cosine(suite[method].T(e), full_credit))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(args.delay_grid, values["direct"], marker="o", label="Direct / TBPTT")
    ax.plot(args.delay_grid, values["dw"], marker="o", label="DW-BP: exact W")
    ax.plot(args.delay_grid, values["dj"], marker="o", label="DJ-BP: exact J")
    ax.set_xlabel("Sequence length / delay T")
    ax.set_ylabel("Credit alignment")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    path = outdir / "credit_alignment_vs_delay.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path, {"plot_credit_alignment_vs_delay": time.perf_counter() - t0}


def make_training_problem(args):
    rng = np.random.default_rng(args.seed + 400)
    W_teacher = make_W(args.n, args.teacher_lambda, args.bulk_radius, rng, args.device)
    W_init = make_W(args.n, args.student_lambda, args.bulk_radius, rng, args.device)
    W_init = 0.65 * W_init + 0.35 * W_teacher + args.init_noise * torch.randn_like(W_init)
    spectral_clip_(W_init, args.train_spectral_clip)
    c = torch.tensor(rng.normal(size=args.n), dtype=torch.float32, device=args.device)
    c = c / c.norm()
    h0 = args.h0_scale * torch.tensor(rng.normal(size=(args.train_batch, args.n)), dtype=torch.float32, device=args.device)
    target = simulate(W_teacher, args.alpha, h0, args.train_T)[:, -1] @ c
    return W_init, c, h0, target.detach()


def training_grad(W, c, h0, target, alpha, T, method, truncation):
    h = simulate(W, alpha, h0, T)
    y = h[:, -1] @ c
    err = y - target
    loss = 0.5 * (err ** 2).mean()
    e = torch.zeros_like(h)
    e[:, -1] = err[:, None] * c[None, :] / h0.shape[0]
    suite = operator_suite(W, h, h0, alpha, truncation, {"max_iter": T + 1, "tol": 1e-10, "early_stop": False})
    credit = suite[method].T(e)
    implicit_grad = suite["constraint"].param_jac().T(credit)["W"]
    return float(loss), -implicit_grad


def train_with_method(args, problem, method, truncation):
    W, c, h0, target = problem
    W = nn.Parameter(W.detach().clone())
    losses = []
    for step in range(args.train_steps + 1):
        loss, grad = training_grad(W, c, h0, target, args.alpha, args.train_T, method, truncation)
        losses.append(loss)
        if step == args.train_steps:
            break
        with torch.no_grad():
            gnorm = grad.norm()
            if gnorm > args.grad_clip:
                grad = grad * (args.grad_clip / gnorm)
            W -= args.lr * grad
            spectral_clip_(W, args.train_spectral_clip)
    return np.asarray(losses)


def plot_training_loss_by_operator(args, outdir):
    t0 = time.perf_counter()
    problem = make_training_problem(args)
    methods = [("full", "Full BPTT"), ("direct", f"Direct k={args.budget_k}"),
               ("dw", f"DW-BP k={args.budget_k}"), ("dj", f"DJ-BP k={args.budget_k}")]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    timings = {}
    for method, label in tqdm(methods, desc="training methods", leave=False):
        start = time.perf_counter()
        losses = train_with_method(args, problem, method, args.budget_k)
        timings[f"train_{method}"] = time.perf_counter() - start
        ax.semilogy(np.arange(len(losses)), losses, label=label)
    ax.set_xlabel("Training step")
    ax.set_ylabel("MSE loss")
    ax.set_title("Training with split kpflow credit operators")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    path = outdir / "training_loss_by_operator.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    timings["plot_training_loss_by_operator"] = time.perf_counter() - t0
    return path, timings


def add_args(p):
    p.add_argument("--out", type=str, default="figures_operator")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--T", type=int, default=24)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--lambda-dom", type=float, default=1.05)
    p.add_argument("--alpha", type=float, default=0.02)
    p.add_argument("--bulk-radius", type=float, default=0.65)
    p.add_argument("--h0-scale", type=float, default=0.05)
    p.add_argument("--max-k", type=int, default=7)
    p.add_argument("--budget-k", type=int, default=3)
    p.add_argument("--delay-grid", type=int, nargs="+", default=[8, 12, 16, 20, 24])
    p.add_argument("--train-T", type=int, default=14)
    p.add_argument("--train-batch", type=int, default=4)
    p.add_argument("--train-steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--teacher-lambda", type=float, default=1.06)
    p.add_argument("--student-lambda", type=float, default=0.94)
    p.add_argument("--init-noise", type=float, default=0.03)
    p.add_argument("--train-spectral-clip", type=float, default=1.25)
    return p


def finalize_args(args):
    args.device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.quick:
        args.T = min(args.T, 14)
        args.n = min(args.n, 4)
        args.max_k = min(args.max_k, 5)
        args.budget_k = min(args.budget_k, args.max_k)
        args.delay_grid = [6, 8, 10, 12]
        args.train_T = min(args.train_T, 10)
        args.train_batch = min(args.train_batch, 3)
        args.train_steps = min(args.train_steps, 4)
    return args


def run(args):
    torch.manual_seed(args.seed)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    all_timings = {}
    paths = []
    print(f"writing figures to {outdir}")
    print(f"settings: device={args.device}, n={args.n}, T={args.T}, max_k={args.max_k}, train_T={args.train_T}")
    for fn in [plot_operator_error_vs_order, plot_truncation_sweep_credit, plot_credit_alignment_vs_delay]:
        path, timings = fn(args, outdir)
        paths.append(path)
        all_timings.update(timings)
        print(f"saved {path}")
    if not args.skip_training:
        path, timings = plot_training_loss_by_operator(args, outdir)
        paths.append(path)
        all_timings.update(timings)
        print(f"saved {path}")
    return paths, all_timings


def print_timing_report(timings):
    print("\nTiming report:")
    for key, val in sorted(timings.items()):
        print(f"  {key:36s} {val:8.3f}s")


def main():
    args = finalize_args(add_args(argparse.ArgumentParser()).parse_args())
    paths, timings = run(args)
    print_timing_report(timings)
    print("\nDone. Saved:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
