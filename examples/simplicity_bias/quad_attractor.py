# quad_fp_rnn_ntk.py
# Four-attractor RNN task:
#   input z=(x,y) sets h0 only
#   autonomous dynamics h_{t+1} = (1-dt) h_t + dt tanh(W h_t + b)
#   output should converge to sign(x), sign(y)
#
# Tracks:
#   loss
#   output accuracy
#   terminal speed ||h_T - h_{T-1}||
#   kpflow HiddenNTKOperator effrank, if installed
#   PCA trajectories
#   phase portraits / vector field in PCA plane
#   final attractor scatter

import os
import math
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

# Optional kpflow NTK.
try:
    from kpflow.grad_op import HiddenNTKOperator
    HAS_KPFLOW = True
except Exception as e:
    print("[warn] kpflow not available; NTK effrank will be skipped.")
    print("       error:", repr(e))
    HAS_KPFLOW = False


# -------------------------
# Config
# -------------------------

SEED = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N = 64
T = 40
DT = 0.20

N_TRAIN = 4096
N_TEST = 1024
BATCH = 256

EPOCHS = 500
LR = 1e-3
WEIGHT_DECAY = 1e-5

NTK_EVERY = 5
NTK_BATCH = 50          # keep small; kpflow can get expensive
PLOT_DIR = "quad_fp_plots"

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(PLOT_DIR, exist_ok=True)


# -------------------------
# Data
# -------------------------

def make_quad_data(n, margin=0.10, noise=0.00, device="cpu"):
    """
    Continuous 2D inputs z=(x,y) in [-1,1]^2.
    Exclude points too close to axes so labels are not ambiguous.
    Target is quadrant sign: (-1,-1), (-1,1), (1,-1), (1,1).
    """
    xs = []
    while len(xs) < n:
        z = torch.empty(n * 2, 2).uniform_(-1.0, 1.0)
        keep = (z[:, 0].abs() > margin) & (z[:, 1].abs() > margin)
        z = z[keep]
        xs.append(z)
        if sum(x.shape[0] for x in xs) >= n:
            break

    x = torch.cat(xs, dim=0)[:n]
    if noise > 0:
        x = x + noise * torch.randn_like(x)

    y = torch.sign(x)
    y[y == 0] = 1.0
    return x.to(device), y.to(device)


x_train, y_train = make_quad_data(N_TRAIN, device=DEVICE)
x_test, y_test = make_quad_data(N_TEST, device=DEVICE)


# -------------------------
# Model
# -------------------------

class SkipRNN(nn.Module):
    """
    Autonomous RNN with input-dependent initial condition only.

    h0 = tanh(W_in x + b_in)
    h_{t+1} = (1-dt) h_t + dt tanh(W_rec h_t + b_rec)
    y_t = W_out h_t + b_out

    The skip dt makes dynamics much more stable than a raw RNN.
    """
    def __init__(self, n=64, dt=0.2, in_dim=2, out_dim=2, gain=1.25):
        super().__init__()
        self.n = n
        self.dt = dt

        self.inp = nn.Linear(in_dim, n)
        self.rec = nn.Linear(n, n)
        self.out = nn.Linear(n, out_dim)

        # Stable-ish initialization, but with enough gain to allow interesting geometry.
        nn.init.normal_(self.inp.weight, std=0.8 / math.sqrt(in_dim))
        nn.init.zeros_(self.inp.bias)

        nn.init.orthogonal_(self.rec.weight)
        with torch.no_grad():
            self.rec.weight.mul_(gain)
        nn.init.zeros_(self.rec.bias)

        nn.init.normal_(self.out.weight, std=0.3 / math.sqrt(n))
        nn.init.zeros_(self.out.bias)

    def forward(self, x, T=T, return_all=True):
        """
        x: [B, 2]
        returns:
            y_seq: [B, T+1, 2]
            h_seq: [B, T+1, N]
        """
        h = torch.tanh(self.inp(x))
        hs = [h]
        ys = [self.out(h)]

        for _ in range(T):
            f = torch.tanh(self.rec(h))
            h = (1.0 - self.dt) * h + self.dt * f
            hs.append(h)
            ys.append(self.out(h))

        h_seq = torch.stack(hs, dim=1)
        y_seq = torch.stack(ys, dim=1)

        if return_all:
            return y_seq, h_seq
        return y_seq[:, -1], h_seq[:, -1]


class GetHidden(nn.Module):
    """
    Wrapper exactly in the style you gave:

        ntk = HiddenNTKOperator(GetHidden(model), inputs, hidden)
        effrank = ntk.effrank()

    Note: model(x)[1] returns hidden trajectory.
    """
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)[1]


model = SkipRNN(n=N, dt=DT).to(DEVICE)
opt = torch.optim.SGD(model.parameters(), lr=LR)#, momentum=0.9, weight_decay=WEIGHT_DECAY)


# -------------------------
# Metrics
# -------------------------

@torch.no_grad()
def eval_metrics(model, x, y):
    model.eval()
    y_seq, h_seq = model(x)
    y_final = y_seq[:, -1]

    loss = torch.mean((y_final - y) ** 2).item()

    pred = torch.sign(y_final)
    pred[pred == 0] = 1
    acc = torch.mean((pred == y).all(dim=1).float()).item()

    terminal_speed = torch.mean(torch.norm(h_seq[:, -1] - h_seq[:, -2], dim=-1)).item()
    output_speed = torch.mean(torch.norm(y_seq[:, -1] - y_seq[:, -2], dim=-1)).item()

    # How spread-out are final hidden states within each quadrant?
    # Small within-class spread means more fixed-point-like bucket formation.
    spreads = []
    for sx in [-1.0, 1.0]:
        for sy in [-1.0, 1.0]:
            mask = (y[:, 0] == sx) & (y[:, 1] == sy)
            if mask.sum() > 2:
                hf = h_seq[mask, -1]
                spreads.append(torch.mean(torch.norm(hf - hf.mean(dim=0), dim=-1)).item())
    hidden_bucket_spread = float(np.mean(spreads)) if spreads else np.nan

    return {
        "loss": loss,
        "acc": acc,
        "terminal_speed": terminal_speed,
        "output_speed": output_speed,
        "hidden_bucket_spread": hidden_bucket_spread,
    }


def compute_ntk_effrank(model, x_probe):
    """
    Uses your kpflow style.

    Depending on the kpflow version, HiddenNTKOperator sometimes wants
    the hidden tensor computed from the same model/input.
    """
    if not HAS_KPFLOW:
        return np.nan

    model.eval()
    try:
        with torch.no_grad():
            _, hidden = model(x_probe)

        ntk = HiddenNTKOperator(GetHidden(model), x_probe, hidden, dev = DEVICE)
        effrank = ntk.effrank()

        if torch.is_tensor(effrank):
            effrank = effrank.detach().cpu().item()
        else:
            effrank = float(effrank)
        return effrank

    except Exception as e:
        print("[warn] NTK computation failed:", repr(e))
        return np.nan


# -------------------------
# PCA helpers
# -------------------------

@torch.no_grad()
def get_pca_basis(model, x, n_components=2):
    """
    PCA basis from all hidden states across batch and time.
    """
    model.eval()
    _, h_seq = model(x)
    H = h_seq.reshape(-1, h_seq.shape[-1])
    H_mean = H.mean(dim=0, keepdim=True)
    Hc = H - H_mean

    # torch.pca_lowrank is usually quick for this size.
    U, S, V = torch.pca_lowrank(Hc, q=n_components)
    basis = V[:, :n_components]
    return H_mean.squeeze(0), basis


@torch.no_grad()
def project_h(h, mean, basis):
    return (h - mean) @ basis


@torch.no_grad()
def pca_trajectories(model, x, mean, basis):
    _, h_seq = model(x)
    B, TT, Nn = h_seq.shape
    z = project_h(h_seq.reshape(-1, Nn), mean, basis).reshape(B, TT, -1)
    return z, h_seq


# -------------------------
# Plotting
# -------------------------

def colors_from_targets(y):
    y_cpu = y.detach().cpu()
    colors = []
    labels = []
    for a, b in y_cpu.tolist():
        if a < 0 and b < 0:
            colors.append("tab:blue")
            labels.append("(-1,-1)")
        elif a < 0 and b > 0:
            colors.append("tab:orange")
            labels.append("(-1,1)")
        elif a > 0 and b < 0:
            colors.append("tab:green")
            labels.append("(1,-1)")
        else:
            colors.append("tab:red")
            labels.append("(1,1)")
    return colors, labels


def plot_training_curves(hist):
    steps = np.array(hist["step"])

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.plot(steps, hist["loss"])
    plt.xlabel("SGD step")
    plt.ylabel("test MSE")
    plt.title("Loss")

    plt.subplot(2, 3, 2)
    plt.plot(steps, hist["acc"])
    plt.xlabel("SGD step")
    plt.ylabel("quadrant accuracy")
    plt.ylim(0, 1.05)
    plt.title("Accuracy")

    plt.subplot(2, 3, 3)
    plt.plot(steps, hist["terminal_speed"])
    plt.xlabel("SGD step")
    plt.ylabel(r"mean $\|h_T-h_{T-1}\|$")
    plt.title("Terminal hidden speed")

    plt.subplot(2, 3, 4)
    plt.plot(steps, hist["output_speed"])
    plt.xlabel("SGD step")
    plt.ylabel(r"mean $\|y_T-y_{T-1}\|$")
    plt.title("Terminal output speed")

    plt.subplot(2, 3, 5)
    plt.plot(steps, hist["hidden_bucket_spread"])
    plt.xlabel("SGD step")
    plt.ylabel("within-quadrant final hidden spread")
    plt.title("Attractor bucket spread")

    plt.subplot(2, 3, 6)
    plt.plot(steps, hist["ntk_effrank"], marker="o", markersize=3)
    plt.xlabel("SGD step")
    plt.ylabel("kpflow HiddenNTK effrank")
    plt.title("Hidden NTK effective rank")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "training_curves.png")
    plt.savefig(path, dpi=180)
    print("saved", path)


def plot_output_map(model, grid_n=80):
    model.eval()
    xs = torch.linspace(-1, 1, grid_n, device=DEVICE)
    ys = torch.linspace(-1, 1, grid_n, device=DEVICE)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    with torch.no_grad():
        y_final, _ = model(pts, return_all=False)
        out = y_final.reshape(grid_n, grid_n, 2)

    out_np = out.detach().cpu().numpy()
    mag = np.linalg.norm(out_np, axis=-1)

    plt.figure(figsize=(13, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(out_np[:, :, 0], extent=[-1, 1, -1, 1], origin="lower", aspect="equal")
    plt.colorbar()
    plt.xlabel("input x")
    plt.ylabel("input y")
    plt.title("final output dim 1")

    plt.subplot(1, 3, 2)
    plt.imshow(out_np[:, :, 1], extent=[-1, 1, -1, 1], origin="lower", aspect="equal")
    plt.colorbar()
    plt.xlabel("input x")
    plt.ylabel("input y")
    plt.title("final output dim 2")

    plt.subplot(1, 3, 3)
    plt.imshow(mag, extent=[-1, 1, -1, 1], origin="lower", aspect="equal")
    plt.colorbar()
    plt.xlabel("input x")
    plt.ylabel("input y")
    plt.title(r"$\|y_T\|$")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "output_map.png")
    plt.savefig(path, dpi=180)
    print("saved", path)


def plot_pca_dynamics(model, x_probe, y_probe, title_suffix="final"):
    model.eval()
    mean, basis = get_pca_basis(model, x_probe)
    z, h_seq = pca_trajectories(model, x_probe, mean, basis)

    z_np = z.detach().cpu().numpy()
    colors, labels = colors_from_targets(y_probe)

    plt.figure(figsize=(8, 7))

    used = set()
    for i in range(z_np.shape[0]):
        lab = labels[i]
        kwargs = {}
        if lab not in used:
            kwargs["label"] = lab
            used.add(lab)

        plt.plot(
            z_np[i, :, 0],
            z_np[i, :, 1],
            color=colors[i],
            alpha=0.25,
            linewidth=1.0,
            **kwargs,
        )
        plt.scatter(z_np[i, 0, 0], z_np[i, 0, 1], color=colors[i], s=8, alpha=0.45)
        plt.scatter(z_np[i, -1, 0], z_np[i, -1, 1], color=colors[i], s=22, alpha=0.9)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA hidden trajectories: {title_suffix}\nsmall=start, large=end")
    plt.legend(frameon=False)
    plt.axis("equal")
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, f"pca_trajectories_{title_suffix}.png")
    plt.savefig(path, dpi=180)
    print("saved", path)

    return mean, basis


def plot_pca_phase_field(model, mean, basis, lim=None, grid_n=25):
    """
    Visualizes learned autonomous dynamics projected into PCA plane.

    For each PCA coordinate z, lift to hidden h = mean + z1 pc1 + z2 pc2,
    apply one autonomous RNN step, then project back.
    """
    model.eval()

    if lim is None:
        lim = 3.0

    zs1 = torch.linspace(-lim, lim, grid_n, device=DEVICE)
    zs2 = torch.linspace(-lim, lim, grid_n, device=DEVICE)
    Z1, Z2 = torch.meshgrid(zs1, zs2, indexing="xy")
    Z = torch.stack([Z1.reshape(-1), Z2.reshape(-1)], dim=1)

    with torch.no_grad():
        h = mean[None, :] + Z @ basis.T
        f = torch.tanh(model.rec(h))
        h_next = (1.0 - model.dt) * h + model.dt * f
        Z_next = project_h(h_next, mean, basis)
        dZ = Z_next - Z

    Z_np = Z.detach().cpu().numpy()
    dZ_np = dZ.detach().cpu().numpy()
    speed = np.linalg.norm(dZ_np, axis=1)

    plt.figure(figsize=(7, 7))
    plt.quiver(
        Z_np[:, 0],
        Z_np[:, 1],
        dZ_np[:, 0],
        dZ_np[:, 1],
        speed,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.003,
        alpha=0.8,
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Projected autonomous vector field in PCA plane")
    plt.axis("equal")
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, "pca_phase_field.png")
    plt.savefig(path, dpi=180)
    print("saved", path)


def plot_snapshots(snapshot_data):
    """
    PCA trajectory snapshots over training, all projected into the final model PCA basis.
    """
    if len(snapshot_data) == 0:
        return

    ncols = min(4, len(snapshot_data))
    nrows = math.ceil(len(snapshot_data) / ncols)

    plt.figure(figsize=(4.3 * ncols, 4.1 * nrows))

    for k, item in enumerate(snapshot_data):
        step = item["step"]
        z_np = item["z"]
        colors = item["colors"]
        labels = item["labels"]

        plt.subplot(nrows, ncols, k + 1)
        used = set()
        for i in range(z_np.shape[0]):
            lab = labels[i]
            kwargs = {}
            if lab not in used:
                kwargs["label"] = lab
                used.add(lab)

            plt.plot(z_np[i, :, 0], z_np[i, :, 1], color=colors[i], alpha=0.20, linewidth=0.8, **kwargs)
            plt.scatter(z_np[i, -1, 0], z_np[i, -1, 1], color=colors[i], s=8, alpha=0.7)

        plt.title(f"step {step}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.axis("equal")

    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(handles, labels, frameon=False, loc="best")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "pca_snapshots.png")
    plt.savefig(path, dpi=180)
    print("saved", path)


# -------------------------
# Training
# -------------------------

hist = {
    "step": [],
    "loss": [],
    "acc": [],
    "terminal_speed": [],
    "output_speed": [],
    "hidden_bucket_spread": [],
    "ntk_effrank": [],
}

# Fixed probe set for plots / NTK.
x_probe, y_probe = make_quad_data(256, margin=0.05, device=DEVICE)
x_ntk, y_ntk = make_quad_data(NTK_BATCH, margin=0.15, device=DEVICE)

snapshot_steps = {0, 25, 75, 150, 250, EPOCHS}
raw_snapshots = []

def record(step):
    m = eval_metrics(model, x_test, y_test)

    if step % NTK_EVERY == 0 or step in snapshot_steps:
        effrank = compute_ntk_effrank(model, x_ntk)
    else:
        effrank = np.nan

    hist["step"].append(step)
    hist["loss"].append(m["loss"])
    hist["acc"].append(m["acc"])
    hist["terminal_speed"].append(m["terminal_speed"])
    hist["output_speed"].append(m["output_speed"])
    hist["hidden_bucket_spread"].append(m["hidden_bucket_spread"])
    hist["ntk_effrank"].append(effrank)

    print(
        f"step {step:04d} | "
        f"loss {m['loss']:.4e} | "
        f"acc {m['acc']:.3f} | "
        f"speed {m['terminal_speed']:.3e} | "
        f"spread {m['hidden_bucket_spread']:.3e} | "
        f"NTK erank {effrank:.3f}"
    )


record(0)

for step in range(1, EPOCHS + 1):
    model.train()

    idx = torch.randint(0, N_TRAIN, (BATCH,), device=DEVICE)
    xb = x_train[idx]
    yb = y_train[idx]

    y_seq, h_seq = model(xb)
    y_final = y_seq[:, -1]

    # Main terminal target.
    terminal_loss = torch.mean((y_final - yb) ** 2)

    # Mild output stabilization over last few steps.
    # This encourages actual fixed-point behavior, not just final-time matching.
    tail = y_seq[:, -8:]
    tail_target = yb[:, None, :].expand_as(tail)
    tail_loss = torch.mean((tail - tail_target) ** 2)

    # Mild hidden terminal speed penalty.
    speed_loss = torch.mean((h_seq[:, -1] - h_seq[:, -2]) ** 2)

    loss = terminal_loss + 0.25 * tail_loss + 0.05 * speed_loss

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()

    if step % 10 == 0 or step in snapshot_steps:
        record(step)

    # Store raw snapshot model states cheaply by storing h trajectories now.
    # Later we re-project into final PCA basis.
    if step in snapshot_steps:
        with torch.no_grad():
            _, htmp = model(x_probe)
            raw_snapshots.append({
                "step": step,
                "h": htmp.detach().cpu(),
                "colors": colors_from_targets(y_probe)[0],
                "labels": colors_from_targets(y_probe)[1],
            })


# -------------------------
# Final plots
# -------------------------

plot_training_curves(hist)
plot_output_map(model)

mean, basis = plot_pca_dynamics(model, x_probe, y_probe, title_suffix="final")

# Choose phase-field limit from final projected trajectories.
with torch.no_grad():
    z_final, _ = pca_trajectories(model, x_probe, mean, basis)
lim = float(torch.quantile(z_final.abs(), 0.98).detach().cpu()) * 1.25
lim = max(lim, 1.0)
plot_pca_phase_field(model, mean, basis, lim=lim, grid_n=27)

# Re-project snapshots into final PCA basis.
snapshot_data = []
for item in raw_snapshots:
    h = item["h"].to(DEVICE)
    Bp, TTp, Np = h.shape
    with torch.no_grad():
        z = project_h(h.reshape(-1, Np), mean, basis).reshape(Bp, TTp, -1)
    snapshot_data.append({
        "step": item["step"],
        "z": z.detach().cpu().numpy(),
        "colors": item["colors"],
        "labels": item["labels"],
    })

plot_snapshots(snapshot_data)
plt.show()

print("\nDone. Plots saved in:", PLOT_DIR)
