import math
import os
import json
from dataclasses import dataclass, replace, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm

import sys
sys.path.append('../../')
from common import (
    project,
    plot_trajectories,
    compute_svs,
    set_mpl_defaults,
    plot_traj_mempro,
    imshow_nonuniform,
    effdim,
    skree_plot,
)


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    # sequence/data
    seq_len: int = 40
    input_dim: int = 16
    input_rank: int = 4
    output_dim: int = 1
    input_scale: float = 1.0
    target_noise_std: float = 0.0

    # shared teacher/student architecture
    hidden_dim: int = 64
    g: float = 1.0                    # student recurrent scaling g_0
    teacher_g: float = 1.2            # teacher recurrent scaling g^*
    train_case: str = "W"              # one of {"W", "Win", "both"}

    # optimization
    batch_size: int = 128
    num_train_steps: int = 100000
    eval_every: int = 250
    lr: float = 1e-2
    momentum: float = 0.0
    weight_decay: float = 0.0
    grad_clip: float | None = None

    # reproducibility/device
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_float_for_name(x: float) -> str:
    s = f"{x:.6g}"
    return s.replace("-", "m").replace(".", "p")


def make_run_name(cfg: Config) -> str:
    run_name = (
        f"case={cfg.train_case}, g={cfg.g:.3f}, teacher_g={cfg.teacher_g:.3f}, "
        f"input_rank={cfg.input_rank}, seed={cfg.seed}"
    )
    safe_name = run_name.replace(" ", "_").replace(",", "").replace("=", "")
    return safe_name


# ============================================================
# Student: vanilla tanh RNN
# ============================================================

class VanillaRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, g: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.W_in = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_out = nn.Linear(hidden_dim, output_dim, bias=True)

        self.reset_parameters(g=g)

    def reset_parameters(self, g: float = 1.0):
        nn.init.xavier_normal_(self.W_in.weight, gain=1.0)
        nn.init.zeros_(self.W_in.bias)

        nn.init.orthogonal_(self.W_h.weight, gain=1.0)
        with torch.no_grad():
            self.W_h.weight.mul_(g)

        nn.init.xavier_normal_(self.W_out.weight, gain=1.0)
        nn.init.zeros_(self.W_out.bias)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        """
        x: (B, T, d)
        returns:
            yhat: (B, T, p)
            hs:   (B, T, H)
        """
        B, T, _ = x.shape
        H = self.hidden_dim

        if h0 is None:
            h = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        else:
            h = h0

        ys = []
        hs = []

        for t in range(T):
            h = torch.tanh(self.W_in(x[:, t]) + self.W_h(h))
            y = self.W_out(h)
            hs.append(h)
            ys.append(y)

        return torch.stack(ys, dim=1), torch.stack(hs, dim=1)


# ============================================================
# Teacher-student task with shared base matrix
# ============================================================

class TeacherStudentTask:
    """
    Teacher-student setup with three trainable-block cases:
      - train_case="W":    teacher/student differ only in recurrent weights
      - train_case="Win":  teacher/student differ only in input weights
      - train_case="both": teacher/student differ in both

    The evaluation code is left unchanged and always inspects the same model/task API.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = cfg.device

        reference = VanillaRNN(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim,
            g=1.0,
        ).to(cfg.device)

        with torch.no_grad():
            self.base_W_h = reference.W_h.weight.detach().clone()
            self.teacher_W_in_weight = reference.W_in.weight.detach().clone()
            self.teacher_W_in_bias = reference.W_in.bias.detach().clone()
            self.shared_W_out_weight = reference.W_out.weight.detach().clone()
            self.shared_W_out_bias = reference.W_out.bias.detach().clone()

        student_input_reference = VanillaRNN(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim,
            g=1.0,
        ).to(cfg.device)
        with torch.no_grad():
            self.student_W_in_weight = student_input_reference.W_in.weight.detach().clone()
            self.student_W_in_bias = student_input_reference.W_in.bias.detach().clone()

        self.teacher = VanillaRNN(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim,
            g=1.0,
        ).to(cfg.device)

        with torch.no_grad():
            self.teacher.W_in.weight.copy_(self.teacher_W_in_weight)
            self.teacher.W_in.bias.copy_(self.teacher_W_in_bias)
            self.teacher.W_out.weight.copy_(self.shared_W_out_weight)
            self.teacher.W_out.bias.copy_(self.shared_W_out_bias)
            self.teacher.W_h.weight.copy_(cfg.teacher_g * self.base_W_h)

        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        rank = min(cfg.input_rank, cfg.input_dim)
        basis, _ = torch.linalg.qr(torch.randn(cfg.input_dim, rank, device=cfg.device), mode="reduced")
        self.input_basis = basis
        self.effective_input_rank = rank

    def initialize_student(self, model: VanillaRNN) -> None:
        case = self.cfg.train_case
        with torch.no_grad():
            if case == "W":
                model.W_in.weight.copy_(self.teacher_W_in_weight)
                model.W_in.bias.copy_(self.teacher_W_in_bias)
#                model.W_h.weight.copy_(self.cfg.g * self.base_W_h)
                nn.init.orthogonal_(model.W_h.weight, gain=1.0)
                model.W_h.weight.mul_(self.cfg.g)
            elif case == "Win":
                model.W_in.weight.copy_(self.student_W_in_weight)
                model.W_in.bias.copy_(self.student_W_in_bias)
                model.W_in.weight.mul_(self.cfg.g)
                model.W_h.weight.copy_(self.cfg.teacher_g * self.base_W_h)
            elif case == "both":
                model.W_in.weight.copy_(self.student_W_in_weight)
                model.W_in.bias.copy_(self.student_W_in_bias)
                model.W_in.weight.mul_(self.cfg.g)
#                model.W_h.weight.copy_(self.cfg.g * self.base_W_h)
                nn.init.orthogonal_(model.W_h.weight, gain=1.0)
                model.W_h.weight.mul_(self.cfg.g)
            else:
                raise ValueError(f"Unknown train_case={case!r}; expected one of ['W', 'Win', 'both']")

            model.W_out.weight.copy_(self.shared_W_out_weight)
            model.W_out.bias.copy_(self.shared_W_out_bias)

        model.W_in.weight.requires_grad_(case in {"Win", "both"})
        model.W_in.bias.requires_grad_(case in {"Win", "both"})
        model.W_h.weight.requires_grad_(case in {"W", "both"})
        model.W_out.weight.requires_grad_(False)
        model.W_out.bias.requires_grad_(False)

    def _sample_low_rank_inputs(self, batch_size: int) -> torch.Tensor:
        z = torch.randn(
            batch_size,
            self.cfg.seq_len,
            self.effective_input_rank,
            device=self.cfg.device,
        )
#        z = z / math.sqrt(max(self.effective_input_rank, 1))
        x = torch.einsum("btr,dr->btd", z, self.input_basis)
        x = self.cfg.input_scale * x
        return x

    @torch.no_grad()
    def sample_batch(self, batch_size: int):
        x = self._sample_low_rank_inputs(batch_size)
        y, h_teacher = self.teacher(x)
        if self.cfg.target_noise_std > 0:
            y = y + self.cfg.target_noise_std * torch.randn_like(y)
        return x, y, h_teacher


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model: nn.Module, task: TeacherStudentTask, batch_size: int = 512, eval_stats: bool = False):
    model.eval()
    x, y, h_teacher = task.sample_batch(batch_size)
    yhat, hs = model(x)
    loss = nn.functional.mse_loss(yhat, y)

    if not eval_stats:
        return {"loss": loss.item()}

    H = hs.reshape(-1, hs.shape[-1])
    X = x.reshape(-1, x.shape[-1])
    V = torch.cat((H, X), -1)
    gram = V.T @ V
    V_effrank = torch.trace(gram) ** 2 / torch.trace(gram @ gram.T)

    W = model.W_h.weight.data
    I_W = torch.eye(W.shape[0], device=W.device, dtype=W.dtype) - W
    R = torch.linalg.inv(I_W)
    R_gram = R @ R.T
    R_effrank = torch.trace(R_gram) ** 2 / torch.trace(R_gram @ R_gram.T)

    from kpflow.grad_op import HiddenNTKOperator as NTK

    class GetHidden(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x)[1]

    model_wrap = GetHidden(model)
    ntk = NTK(model_wrap, x, hs, dev=x.device)
    ntk_time = ntk.partial_avg(-1)
    ntk_space = ntk.partial_avg((0, 1))
    ntk_targ_cos = ntk_time.rayleigh_coef(y)

    ntk_space_rank = ntk_space.effrank(nsamp=50)
    ntk_time_rank = ntk_time.effrank(nsamp=50)
    ntk_full_rank = ntk.effrank(nsamp=50)
    ntk_trace = ntk.trace(nsamp=50)

    return {
        "loss": loss.item(),
        "V_effrank": V_effrank.item(),
        "R_effrank": R_effrank.item(),
        "ntk_targ_cos": ntk_targ_cos.item(),
        "ntk_space_rank": ntk_space_rank.item(),
        "ntk_time_rank": ntk_time_rank.item(),
        "ntk_full_rank": ntk_full_rank.item(),
        "ntk_trace": ntk_trace.item(),
    }

def make_name(cfg):
    run_name = (
        f"case={cfg.train_case}, g={cfg.g:.3f}, teacher_g={cfg.teacher_g:.3f}, "
        f"input_rank={cfg.input_rank}, seed={cfg.seed}"
    )
    safe_name = run_name.replace(" ", "_").replace(",", "").replace("=", "")
    return safe_name

# ============================================================
# Training
# ============================================================

def train_and_save(cfg: Config, save_dir: str = "trained_models"):
    set_seed(cfg.seed)

    task = TeacherStudentTask(cfg)
    model = VanillaRNN(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim,
        g=1.0,
    ).to(cfg.device)
    task.initialize_student(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        trainable_params,
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    run_name = (
        f"case={cfg.train_case}, g={cfg.g:.3f}, teacher_g={cfg.teacher_g:.3f}, "
        f"input_rank={cfg.input_rank}, seed={cfg.seed}"
    )
    safe_name = run_name.replace(" ", "_").replace(",", "").replace("=", "")

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{safe_name}_metrics.csv")
    pt_path = os.path.join(save_dir, f"{safe_name}.pt")
    json_path = os.path.join(save_dir, f"{safe_name}.json")

    print("=" * 80)
    print("Teacher/student tanh RNN task")
    print(f"device             : {cfg.device}")
    print(f"seq_len            : {cfg.seq_len}")
    print(f"input_dim          : {cfg.input_dim}")
    print(f"input_rank         : {cfg.input_rank}")
    print(f"hidden_dim         : {cfg.hidden_dim}")
    print(f"student g          : {cfg.g}")
    print(f"teacher g*         : {cfg.teacher_g}")
    print(f"train case         : {cfg.train_case}")
    print(f"optimizer          : SGD")
    print(f"lr                 : {cfg.lr}")
    print("=" * 80)

    train_history = []

    import wandb
    wandb.init(project="teacher_student_mine", config=cfg, name=run_name)

    for step in range(1, cfg.num_train_steps + 1):
        model.train()
        x, y, _ = task.sample_batch(cfg.batch_size)

        optimizer.zero_grad(set_to_none=True)
        yhat, _ = model(x)
        loss = nn.functional.mse_loss(yhat, y)
        loss.backward()

        if cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)

        optimizer.step()

        if step % cfg.eval_every == 0 or step == 1:
            train_metrics = evaluate(model, task, batch_size=cfg.batch_size, eval_stats=False)
            test_metrics = evaluate(model, task, batch_size=512, eval_stats=True)
            rel_targ_overlap = test_metrics["ntk_targ_cos"] / max(test_metrics["ntk_trace"], 1e-12)
            row = {
                "step": step,
                "train_loss": float(train_metrics["loss"]),
                "test_loss": float(test_metrics["loss"]),
                "ntk_space_rank": float(test_metrics["ntk_space_rank"]),
                "ntk_time_rank": float(test_metrics["ntk_time_rank"]),
                "ntk_targ_cos": float(test_metrics["ntk_targ_cos"]),
                "ntk_trace": float(test_metrics["ntk_trace"]),
                "rel_targ_overlap": float(rel_targ_overlap),
                "g": float(cfg.g),
                "teacher_g": float(cfg.teacher_g),
                "input_rank": int(cfg.input_rank),
                "seed": int(cfg.seed),
                "train_case": str(cfg.train_case),
            }
            train_history.append(row)

            print(
                f"step {step:5d} | "
                f"train loss {row['train_loss']:.6f} | "
                f"test loss {row['test_loss']:.6f} | "
                f"time rank {row['ntk_time_rank']:.3f} | "
                f"rel overlap {row['rel_targ_overlap']:.6e}"
            )
            wandb.log(row)

    history_df = pd.DataFrame(train_history)
    history_df.to_csv(csv_path, index=False)

    payload = {
        "model_state_dict": model.state_dict(),
        "cfg": asdict(cfg),
        "history": train_history,
        "csv_path": csv_path,
    }
    torch.save(payload, pt_path)

    with open(json_path, "w") as f:
        json.dump(
            {
                "run_name": run_name,
                "cfg": asdict(cfg),
                "model_path": pt_path,
                "csv_path": csv_path,
            },
            f,
            indent=2,
        )

    wandb.finish()

    return model, task, train_history


# ============================================================
# Compare loaded model to target
# ============================================================

def load_model_and_compare(pt_path, batch_size=16, device=None, plot_example_idx=0):
    payload = torch.load(pt_path, map_location="cpu")

    if "cfg" not in payload:
        raise KeyError(f"Checkpoint missing 'cfg': {pt_path}")
    if "model_state_dict" not in payload:
        raise KeyError(f"Checkpoint missing 'model_state_dict': {pt_path}")

    cfg_dict = dict(payload["cfg"])
    if device is not None:
        cfg_dict["device"] = device
    cfg = Config(**cfg_dict)

    set_seed(cfg.seed)

    task = TeacherStudentTask(cfg)
    model = VanillaRNN(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim,
        g=1.0,
    ).to(cfg.device)
    task.initialize_student(model)

    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    x, y, h_teacher = task.sample_batch(batch_size)
    with torch.no_grad():
        yhat, hs = model(x)

    mse = torch.mean((yhat - y) ** 2).item()
    mae = torch.mean(torch.abs(yhat - y)).item()

    print(f"Loaded checkpoint : {pt_path}")
    print(f"student g         : {cfg.g}")
    print(f"teacher g*        : {cfg.teacher_g}")
    print(f"input_rank        : {cfg.input_rank}")
    print(f"hidden_dim        : {cfg.hidden_dim}")
    print(f"seed              : {cfg.seed}")
    print(f"MSE on batch      : {mse:.6f}")
    print(f"MAE on batch      : {mae:.6f}")
    print(f"x shape           : {tuple(x.shape)}")
    print(f"y shape           : {tuple(y.shape)}")
    print(f"yhat shape        : {tuple(yhat.shape)}")

    i = int(plot_example_idx)
    if i < 0 or i >= batch_size:
        raise IndexError(f"plot_example_idx={i} out of range for batch_size={batch_size}")

    plt.figure()
    plt.plot(x[i].detach().cpu())

    y_i = y[i].detach().cpu()
    yhat_i = yhat[i].detach().cpu()

    if y_i.shape[-1] == 1:
        y_plot = y_i.squeeze(-1).numpy()
        yhat_plot = yhat_i.squeeze(-1).numpy()
    else:
        y_plot = y_i.mean(-1).numpy()
        yhat_plot = yhat_i.mean(-1).numpy()

    err_plot = yhat_plot - y_plot

    df = pd.DataFrame(
        {
            "t": np.arange(len(y_plot)),
            "target": y_plot,
            "prediction": yhat_plot,
            "error": err_plot,
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    axes[0].plot(df["t"], df["target"], label="target")
    axes[0].plot(df["t"], df["prediction"], label="prediction")
    axes[0].set_ylabel("output")
    axes[0].set_title(
        f"Teacher/student RNN | case={cfg.train_case}, g={cfg.g}, g*={cfg.teacher_g}, input_rank={cfg.input_rank}, seed={cfg.seed}"
    )
    axes[0].legend()

    axes[1].plot(df["t"], df["error"], label="prediction - target")
    axes[1].axhline(0.0, linewidth=1)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("error")
    axes[1].legend()

    plt.tight_layout()

    return {
        "cfg": cfg,
        "payload": payload,
        "x": x,
        "y": y,
        "yhat": yhat,
        "hidden": hs,
        "teacher_hidden": h_teacher,
        "mse": mse,
        "mae": mae,
        "df_example": df,
        "fig": fig,
        "axes": axes,
    }

def base_cfg():
    return Config(
        seq_len=40,
        input_dim=16,
        input_rank=6,
        hidden_dim=64,
        g=0.4,          # student g0 (used for W or both cases)
        teacher_g=1.0,  # teacher g*
        train_case="W",
        seed=0,
    )

def plot_regime_training_panels(csv_paths, save_path=None, figsize=(12, 4)):
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    ax_loss, ax_rank, ax_overlap = axes

    colors = [*plt.rcParams['axes.prop_cycle'].by_key()['color']]
    for idx, (name, csv_path) in enumerate(csv_paths.items()):
        df = pd.read_csv(csv_path)
        steps = df["step"].to_numpy()

        ax_loss.plot(steps, df["test_loss"].to_numpy(), lw=2, label=name)
        ax_rank.plot(steps, df["ntk_time_rank"].to_numpy(), lw=2, label=name)
        overlap = df["rel_targ_overlap"].to_numpy()
        mask = overlap < 1e8
        ax_overlap.plot(steps[mask], overlap[mask], lw=2, label=name)

    ax_loss.set_title("(d) Loss over training")
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("test loss")
    ax_loss.set_yscale("log")
    ax_loss.grid(alpha=0.25)

    ax_rank.set_title("(e) Temporal rank over training")
    ax_rank.set_xlabel("step")
    ax_rank.set_ylabel("ntk_time_rank")
    ax_rank.grid(alpha=0.25)

    ax_overlap.set_title("(f) Target overlap over training")
    ax_overlap.set_xlabel("step")
    ax_overlap.set_ylabel("rel_targ_overlap")
    ax_overlap.grid(alpha=0.25)

    ax_loss.legend()#loc="upper center", ncol=max(1, len(csv_paths)), frameon=False)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, axes

def bottom_row_figure():
    g_values = [0.0, 0.5, 1.5, 2.0]
    input_ranks = [5, 15, 30]
    train_cases = ["W", "Win"]
#    seeds = [0,1,2,3,4,5]
    seeds = [0]

    fig, axes = plt.subplots(len(train_cases), 1, figsize=(3., 3 * len(train_cases)))
    if len(train_cases) == 1:
        axes = [axes]

    for ax, train_case in zip(axes, train_cases):
        csv_paths = {}
        for seed in seeds:
            for g in tqdm(g_values):
                for input_rank in input_ranks:
                    cfg = base_cfg()
                    cfg.seed = seed
                    cfg.g = g
                    cfg.input_rank = int(input_rank)
                    cfg.train_case = train_case
#                    if train_case == 'Win': #  Skip some
#                        if g != 0.5:
#                            continue 
#
#                    if train_case == 'W': # Skip some
#                        if not ((g == 0. and input_rank == 5) or (g == 0.5 and input_rank == 30) or (g == 1.5 and input_rank == 30) or (g == 2. and input_rank == 5)):
#                            continue

                    csv_path = os.path.join("trained_models", f"{make_name(cfg)}_metrics.csv")
                    csv_paths[f"g={g},rx={cfg.input_rank}"] = csv_path
                    if os.path.exists(csv_path):
                        continue
                    train_and_save(cfg, save_dir="trained_models")
            

        colors = [*plt.rcParams['axes.prop_cycle'].by_key()['color']]
        for idx, (name, csv_path) in enumerate(csv_paths.items()):
            df = pd.read_csv(csv_path)
            steps = df["step"].to_numpy()
            ax.plot(steps, df["test_loss"].to_numpy(), lw=2, label=name, color=colors[(idx + (6 if train_case == 'Win' else 0)) % len(colors)])

#        ax.set_title(f"{train_case} training")
        ax.set_xlabel("GD Iter.")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_ylabel('Loss (mse)')
        ax.yaxis.set_ticks_position('right')

        ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig('loss_sweep.pdf')
    return fig, axes

def _aggregate_grid(df, metric, agg="mean"):
    group_cols = ["input_rank", "g"]
    if agg == "mean":
        df_plot = df.groupby(group_cols, as_index=False)[metric].mean()
    elif agg == "median":
        df_plot = df.groupby(group_cols, as_index=False)[metric].median()
    else:
        raise ValueError("agg must be 'mean' or 'median'")

    g_vals = np.sort(df_plot["g"].unique())
    rank_vals = np.sort(df_plot["input_rank"].unique())
    pivot = df_plot.pivot(index="input_rank", columns="g", values=metric)
    pivot = pivot.reindex(index=rank_vals, columns=g_vals)
    return pivot.values.astype(float), g_vals, rank_vals

def plot_ntk_init_heatmaps(
    csv_path,
    agg="mean",
    figsize=(12, 9),
    save_path=None,
    g_max=None,
    cmap_space="viridis",
    cmap_time="magma",
    cmap_align="cividis",
    interpolation="bilinear",
    contour=True,
    contour_levels=5,
    contour_color="w",
    contour_linewidth=0.8,
    contour_alpha=0.9,
    log_space=False,
    log_time=False,
    log_align=False,
    log_eps=1e-12,
    use_first_order_approx=False,
    normalize_align=False,
):
    df = pd.read_csv(csv_path)
    required = {"g", "input_rank", "ntk_space_rank", "ntk_time_rank", "ntk_targ_cos", "train_case"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if g_max is not None:
        df = df[df["g"] <= g_max].copy()
    if len(df) == 0:
        raise ValueError("No rows left after filtering by g_max.")

    train_cases = [c for c in ["W", "Win"] if c in set(df["train_case"].unique())]
    if len(train_cases) == 0:
        raise ValueError("No supported train_case values found.")

    def _maybe_log(Z, use_log, name):
        if not use_log:
            return Z, name
        Zp = np.asarray(Z, dtype=float)
        if np.any(Zp < 0):
            raise ValueError(f"{name} has negative values, cannot apply log10 scaling.")
        return np.log10(Zp + log_eps), f"log10({name})"

    fig, axes = plt.subplots(len(train_cases), 3, figsize=figsize)
    out = {}

    for row_idx, train_case in enumerate(train_cases):
        df_case = df[df["train_case"] == train_case].copy()

        Z_space, g_vals, rank_vals = _aggregate_grid(df_case, "ntk_space_rank", agg=agg)
        Z_time, _, _ = _aggregate_grid(df_case, "ntk_time_rank", agg=agg)
        Z_full, _, _ = _aggregate_grid(df_case, "ntk_full_rank", agg=agg)
        Z_align, _, _ = _aggregate_grid(df_case, "ntk_targ_cos", agg=agg)
        ntk_trace, _, _ = _aggregate_grid(df_case, "ntk_trace", agg=agg)

        if use_first_order_approx:
            Z_space, _, _ = _aggregate_grid(df_case, "R_effrank", agg=agg)
            Z_time, _, _ = _aggregate_grid(df_case, "V_effrank", agg=agg)

        if normalize_align:
            Z_align = Z_align / np.maximum(ntk_trace, 1e-12)

        Z_space_plot, _ = _maybe_log(Z_space, log_space, "ntk_space_rank")
        Z_time_plot, _ = _maybe_log(Z_time, log_time, "ntk_time_rank")
        Z_align_plot, _ = _maybe_log(Z_align, log_align, "ntk_targ_cos")
        ntk_trace_plot, _ = _maybe_log(ntk_trace, True, "ntk_trace")

        extent = [g_vals.min(), g_vals.max(), rank_vals.min(), rank_vals.max()]
        G, R = np.meshgrid(g_vals, rank_vals)

        def _draw_panel(ax, Z_plot, cmap, title, ylabel=None):
            im = ax.imshow(
                Z_plot,
                origin="lower",
                aspect="auto",
                extent=extent,
                interpolation=interpolation,
                cmap=cmap,
            )
            if contour:
                try:
                    cs = ax.contour(
                        G, R, Z_plot,
                        levels=contour_levels,
                        colors=contour_color,
                        linewidths=contour_linewidth,
                        alpha=contour_alpha,
                    )
               #     ax.clabel(cs, inline=True, fontsize=7, fmt="%.2g")
                except Exception:
                    pass
            ax.set_title(title)
            ax.set_xlabel(r"Initial Connectivity Scale $g$")
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        row_ylabel = f"{train_case}\nInput Rank $r_x$"
        _draw_panel(axes[row_idx, 0], Z_space_plot, cmap_space, r"NTK$_S$ Spatial Rank", ylabel=row_ylabel)
        _draw_panel(axes[row_idx, 1], Z_time_plot, cmap_time, r"NTK$_S$ Temporal Rank")
        _draw_panel(axes[row_idx, 2], Z_align_plot, cmap_align, ('Log ' if log_align else '') + r"Target-NTK Alignment")
#        _draw_panel(axes[row_idx, 3], ntk_trace_plot, 'copper', r"Log NTK Trace")

        out[train_case] = {
            "g_vals": g_vals,
            "input_ranks": rank_vals,
            "Z_space": Z_space,
            "Z_time": Z_time,
            "Z_full": Z_full,
            "Z_align": Z_align,
            "Z_trace": ntk_trace,
        }

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, axes, out

def _to_python_scalar(x):
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.detach().cpu().item()
        return x.detach().cpu().tolist()
    try:
        return float(x)
    except (TypeError, ValueError):
        return x


def sweep_init_grid(
    g_values,
    input_ranks,
    train_cases=("W", "Win", "both"),
    output_dir="init_sweep",
    batch_size=512,
    seeds=(0,),
    device=None,
):
    os.makedirs(output_dir, exist_ok=True)

    all_rows = []
    full_results = []

    for train_case in train_cases:
        for seed in seeds:
            for input_rank in input_ranks:
                for g in g_values:
                    cfg = base_cfg()
                    cfg.train_case = str(train_case)
                    cfg.input_rank = int(input_rank)
                    cfg.seed = int(seed)
                    cfg.g = float(g)
                    if device is not None:
                        cfg.device = device

                    set_seed(cfg.seed)
                    task = TeacherStudentTask(cfg)
                    model = VanillaRNN(
                        input_dim=cfg.input_dim,
                        hidden_dim=cfg.hidden_dim,
                        output_dim=cfg.output_dim,
                        g=1.0,
                    ).to(cfg.device)
                    task.initialize_student(model)

                    with torch.no_grad():
                        metrics = evaluate(model, task, batch_size=batch_size, eval_stats=True)

                    metrics_clean = {k: _to_python_scalar(v) for k, v in metrics.items()}
                    row = {
                        "seed": seed,
                        "train_case": str(train_case),
                        "g": float(g),
                        "input_rank": int(input_rank),
                        "batch_size": int(batch_size),
                        "device": cfg.device,
                        **metrics_clean,
                    }
                    all_rows.append(row)
                    full_results.append({
                        "cfg": asdict(cfg),
                        "seed": seed,
                        "train_case": str(train_case),
                        "g": float(g),
                        "input_rank": int(input_rank),
                        "metrics": metrics_clean,
                    })

                    print(
                        f"[case={train_case:>4s} | seed={seed:02d} | g={g:.4f} | input_rank={input_rank:3d}] "
                        + " | ".join(
                            f"{k}={v:.4g}"
                            for k, v in metrics_clean.items()
                            if isinstance(v, (int, float))
                        )
                    )

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(output_dir, "init_sweep_results.csv")
    df.to_csv(csv_path, index=False)

    pt_path = os.path.join(output_dir, "init_sweep_results.pt")
    torch.save(full_results, pt_path)

    meta = {
        "base_cfg": asdict(base_cfg()),
        "g_values": list(map(float, g_values)),
        "input_ranks": list(map(int, input_ranks)),
        "train_cases": list(map(str, train_cases)),
        "seeds": list(map(int, seeds)),
        "batch_size": int(batch_size),
        "device": device if device is not None else base_cfg().device,
        "csv_path": csv_path,
        "pt_path": pt_path,
    }
    json_path = os.path.join(output_dir, "init_sweep_metadata.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved CSV results to: {csv_path}")
    print(f"Saved full PT results to: {pt_path}")
    print(f"Saved metadata to: {json_path}")
    return df

def sweep():
    cfg = base_cfg()
    g_values = np.linspace(0.0, 3.0, 50)
    input_ranks = np.arange(2, 63, 4)

    # 30, 5

    df = sweep_init_grid(
        g_values=g_values,
        input_ranks=input_ranks,
        train_cases=("W", "Win", "both"),
        output_dir="init_sweep",
        batch_size=200,
        seeds=[0, 1, 2],
    )
    return df

if __name__ == "__main__":
    set_mpl_defaults(14)

    if not os.path.exists("init_sweep/init_sweep_results.csv"):
        sweep()

    fig, axes, out = plot_ntk_init_heatmaps(
        csv_path="init_sweep/init_sweep_results.csv",
#        g_max=2.6,
        g_max=2.5,
        save_path="init_sweep/ntk_init_heatmaps.pdf",
        cmap_space="viridis",
        cmap_time="magma",
        cmap_align="cividis",
        figsize=(3*4, 2*3),
        contour=True,
        contour_levels=3,
        contour_color="w",
        contour_linewidth=1.5,
        contour_alpha=0.9,
        log_space=False,
        log_time=False,
        log_align=False,   # set True only if your alignment still spans orders of magnitude
        use_first_order_approx=False,
        normalize_align=True
    )
    bottom_row_figure()
    plt.show()
