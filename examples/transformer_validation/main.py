import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import torch, os
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('../')
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


@dataclass
class SimpleAttnMLPConfig:
    d_input: int = 16
    d_model: int = 64
    n_heads: int = 4
    mlp_hidden: int = 128
    mlp_layers: int = 3   # total affine layers in MLP, including final output layer
    d_output: int = 1
    max_seq_len: int = 256
    use_positional_embedding: bool = True

    # Fourier features
    use_fourier_features: bool = False
    fourier_n_freqs: int = 32
    fourier_include_input: bool = True

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: [B, T, D]
        """
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,T,dh]
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)     # [B,H,T,T]
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)                                # [B,H,T,T]
        a = torch.matmul(attn_weights, v)                                           # [B,H,T,dh]
        attn_out = a.transpose(1, 2).contiguous().view(B, T, D)              # [B,T,D]
        attn_out = self.o_proj(attn_out)                                             # [B,T,D]

        if need_weights:
            return attn_out, attn_weights, a
        return attn_out, None, a


class DeepMLP(nn.Module):
    """
    MLP with mlp_layers total affine layers.
    Example:
      mlp_layers = 3 gives
        d_model -> hidden -> hidden -> d_output

    We return all hidden activations excluding final output.
    """
    def __init__(self, d_in: int, d_hidden: int, mlp_layers: int, d_out: int, bias: bool = False):
        super().__init__()
        assert mlp_layers >= 1

        self.mlp_layers = mlp_layers
        self.hidden_linears = nn.ModuleList()

        if mlp_layers == 1:
            self.final_linear = nn.Linear(d_in, d_out, bias = bias)
        else:
            # first hidden layer
            self.hidden_linears.append(nn.Linear(d_in, d_hidden, bias = bias))
            # middle hidden layers
            for _ in range(mlp_layers - 2):
                self.hidden_linears.append(nn.Linear(d_hidden, d_hidden, bias = bias))
            # final output layer
            self.final_linear = nn.Linear(d_hidden, d_out, bias = bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        x: [B, T, D]
        Returns:
          y: [B, T, d_out]
          hidden_activations: list of [B, T, d_hidden]
        """
        hidden_activations = []
        h = x

        for layer in self.hidden_linears:
            pre = F.gelu(h)
            h = layer(pre)
            hidden_activations.append(pre)

        y = self.final_linear(h)
        hidden_activations.append(h)
        return y, hidden_activations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """
    Maps x: [..., d_input] -> [..., d_out]
    using sin/cos random Fourier features.

    If include_input=True, output is [x, sin(2π xB), cos(2π xB)].
    Otherwise output is [sin(2π xB), cos(2π xB)].

    B is fixed after init, so this is a true drop-in feature map.
    """
    def __init__(
        self,
        d_input: int,
        n_freqs: int,
        sigma: float = 1.0,
        include_input: bool = True,
    ):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.sigma = sigma
        self.include_input = include_input

        # Random Gaussian frequency matrix: [d_input, n_freqs]
        B = sigma * torch.randn(d_input, n_freqs)
        self.register_buffer("B", B)

        self.d_out = 2 * n_freqs + (d_input if include_input else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., d_input]
        returns: [..., d_out]
        """
        # [..., n_freqs]
        xb = 2.0 * math.pi * (x @ self.B)

        ff = torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)

        if self.include_input:
            ff = torch.cat([x, ff], dim=-1)

        return ff


class FourierFeatures1DDet(nn.Module):
    def __init__(self, n_freqs, include_input=True, base=2.0, w_min=1.0):
        super().__init__()
        self.include_input = include_input
        freqs = w_min * (base ** torch.arange(n_freqs, dtype=torch.float32))
        self.register_buffer("freqs", freqs)

    @property
    def d_out(self):
        return 2 * len(self.freqs) + (1 if self.include_input else 0)

    def forward(self, x):
        # x: [..., 1]
        z = x * self.freqs.view(*([1] * (x.ndim - 1)), -1)
        ang = 2.0 * math.pi * z
        ff = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if self.include_input:
            ff = torch.cat([x, ff], dim=-1)
        return ff


class SingleBlockAttnMLP(nn.Module):
    """
    input -> input projection -> single self-attention -> deep MLP -> output

    V = cat(attention_output, all hidden MLP activations)
    """
    def __init__(self, cfg: SimpleAttnMLPConfig):
        super().__init__()
        self.cfg = cfg
        if getattr(cfg, "use_fourier_features", False):
            self.fourier = FourierFeatures1DDet(
                n_freqs=cfg.fourier_n_freqs,
                include_input=cfg.fourier_include_input,
            )
            input_dim = self.fourier.d_out
        else:
            self.fourier = None
            input_dim = cfg.d_input

        self.input_proj = nn.Linear(input_dim, cfg.d_model)

        if cfg.use_positional_embedding:
            self.pos_embedding = nn.Parameter(
                0.02 * torch.randn(1, cfg.max_seq_len, cfg.d_model)
            )
        else:
            self.pos_embedding = None

        self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_heads)
        self.mlp = DeepMLP(
            d_in=cfg.d_model,
            d_hidden=cfg.mlp_hidden,
            mlp_layers=cfg.mlp_layers,
            d_out=cfg.d_output,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x: [B, T, d_input]
        Returns:
          y: [B, T, d_output]
          cache:
            input_hidden: [B, T, d_model]
            attn_out: [B, T, d_model]
            mlp_hidden_0, ..., mlp_hidden_{L-2}: hidden activations
            V: concatenation of attn_out and all MLP hidden activations
        """
        B, T, _ = x.shape
        assert T <= self.cfg.max_seq_len

        x_in = self.fourier(x) if self.fourier is not None else x
        h0 = self.input_proj(x_in)  # [B,T,d_model]
        if self.pos_embedding is not None:
            h0 = h0 + self.pos_embedding[:, :T, :]

        attn_out, attn_weights, a = self.attn(h0, attn_mask=attn_mask, need_weights=need_weights)
        y, mlp_hiddens = self.mlp(attn_out)

        V_parts = 3 * [h0] + [a_head for a_head in a.swapaxes(0,1)] + mlp_hiddens
        V = torch.cat(V_parts, dim=-1)

        cache = {
            "input_hidden": h0,
            "attn_out": attn_out,
            "V": V,
            "V_parts": V_parts,
            "x_in": x_in
        }
        for i, h in enumerate(mlp_hiddens):
            cache[f"mlp_hidden_{i}"] = h
        if attn_weights is not None:
            cache["attn_weights"] = attn_weights

        return y, cache


def flatten_bt(z: torch.Tensor) -> torch.Tensor:
    """
    [B, T, D] -> [B*T, D]
    """
    B, T, D = z.shape
    return z.reshape(B * T, D)

def make_temporal_basis(T, num_modes=6):
    """
    Returns orthonormal temporal basis Phi of shape (num_modes, T).

    Mode 0 is approximately constant.
    Higher modes are low/high frequency cosine/sine patterns.
    """
    ts = np.arange(T, dtype=float)
    basis = [np.ones(T)]

    freq = 1
    while len(basis) < num_modes:
        basis.append(np.cos(2 * np.pi * freq * ts / T))
        if len(basis) < num_modes:
            basis.append(np.sin(2 * np.pi * freq * ts / T))
        freq += 1

    M = np.stack(basis[:num_modes], axis=1)   # (T, num_modes)
    Qr, _ = np.linalg.qr(M)                   # orthonormalize columns
    Phi = Qr.T                                # (num_modes, T)
    return Phi


def make_targets_from_input(X, num_modes=6, summary="mean", normalize=True):
    """
    X: input array of shape (B, T) or (B, T, d_in)

    Returns:
      Ys: list of targets, each shape (B, T)
      Phi: temporal basis, shape (num_modes, T)

    Each target is Y_k[b, t] = a_b * phi_k[t],
    where a_b is a scalar summary of input example b.
    """
    B = X.shape[0]
    T = X.shape[1]
    Phi = make_temporal_basis(T, num_modes=num_modes)   # (num_modes, T)

    # scalar amplitude per example
    if X.ndim == 2:
        if summary == "mean":
            a = X.mean(axis=1)                  # (B,)
        elif summary == "first":
            a = X[:, 0]
        elif summary == "last":
            a = X[:, -1]
        else:
            raise ValueError(f"Unknown summary: {summary}")

    elif X.ndim == 3:
        if summary == "mean":
            a = X.mean(axis=(1, 2))             # (B,)
        elif summary == "first":
            a = X[:, 0, :].mean(axis=1)
        elif summary == "last":
            a = X[:, -1, :].mean(axis=1)
        else:
            raise ValueError(f"Unknown summary: {summary}")
    else:
        raise ValueError("X must have shape (B,T) or (B,T,d_in)")

    # optional normalization so all modes have comparable target norm
    if normalize:
        a = a / (np.std(a) + 1e-8)

    Ys = []
    for k in range(num_modes):
        Yk = a[:, None] * Phi[k][None, :]       # (B, T)
        Ys.append(Yk)

    return Ys, Phi


def project_out_mode(Y, Q):
    """
    Project target Y (B,T) to be orthogonal to dominant mode Q (B,T)
    in the usual Euclidean inner product on R^{B x T}.
    """
    alpha = np.sum(Y * Q) / (np.sum(Q * Q) + 1e-12)
    return Y - alpha * Q

def panel_B(cfg):
    os.makedirs("data", exist_ok=True)
    plt.figure()
    d_inputs = np.linspace(1, 1000, 50).astype(int)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for color, B in zip(colors, [5, 10, 15]):
        ranks_ntk_time, ranks_ntk_space, ranks_v, ranks_ntk = [], [], [], []
        ranks_x = []
        if os.path.exists(f'data/ranks_ntk_time_{B}.npy'):
            ranks_ntk_time = np.load(f'data/ranks_ntk_time_{B}.npy')
            ranks_x = np.load(f'data/ranks_x_{B}.npy')
        else:
            for d_input in tqdm(d_inputs):
                cfg.d_input = d_input
                model = SingleBlockAttnMLP(cfg)

                T = 50
                x = torch.randn(B, T, cfg.d_input)

                y, cache = model(x, need_weights=True)

                V_flat = flatten_bt(cache["V"])
                K_core = V_flat @ V_flat.T

                class GetWeightSites(nn.Module):
                    def __init__(self, net):
                        super().__init__()
                        self.net = net

                    def forward(self, x):
                        return self.net(x)[1]["V"] # Relevant quantities.

                from kpflow.grad_op import HiddenNTKOperator as NTK
                model_wrap = GetWeightSites(model)
                vs = model_wrap(x)
                ntk = NTK(model_wrap, x, vs, dev=x.device)
                ntk_time = ntk.partial_avg(-1)

                ranks_ntk_time.append(ntk_time.effrank(nsamp=100, grammian = False))
                ranks_x.append(effdim(cache["x_in"]))
    #            ranks_ntk_space.append(ntk.effrank(keep_dims = -1, nsamp = 20, grammian = False))
    #            ranks_ntk.append(ntk.effrank(nsamp = 20, grammian = False))

        np.save('data/d_inputs.npy', np.array(d_inputs))
        np.save(f'data/ranks_ntk_time_{B}.npy', np.array(ranks_ntk_time))
        np.save(f'data/ranks_x_{B}.npy', np.array(ranks_x))
#        plt.subplot(1,2,1)
        plt.plot(d_inputs, ranks_ntk_time, label = f'$n_x = {B}$', color = color)
        plt.plot(d_inputs, ranks_x, label = f'$n_x = {B}$', color = color, linestyle = 'dashed')
        plt.xlabel('Input Dim')
        plt.title('NTK Temporal Rank')
#        plt.subplot(1,2,2)
#        plt.plot(d_inputs, ranks_ntk_space)
#        plt.xlabel('Input Dim')
#        plt.title('NTK Spatial Rank')

#    plt.subplot(1,2,1)
    plt.legend(ncol = 3)
    plt.savefig('panel_B.pdf')

def panel_C(cfg, B = 10):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure(figsize = (1.1*4, 1.1*3 * 2))
    for plot_idx, d_input in enumerate([1, 5, 100]):
        cfg.d_input = d_input
        model = SingleBlockAttnMLP(cfg)

        T = 50
        x = torch.randn(B, T, cfg.d_input)
        y, cache = model(x, need_weights=True)

        V_flat = flatten_bt(cache["V"])
        K_core = V_flat @ V_flat.T

        class GetWeightSites(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, x):
                return self.net(x)[1]["V"] # Relevant quantities.

        from kpflow.grad_op import HiddenNTKOperator as NTK
        model_wrap = GetWeightSites(model)
        vs = model_wrap(x)
        ntk = NTK(model_wrap, x, vs, dev=x.device)
        ntk_time = ntk.partial_avg(-1)

        svals, svecs = ntk_time.svd(10, grammian = False, compute_vecs = True)
        Q = svecs[0, :, :, 0]

        print(ntk_time.effdim())

        plt.subplot(3,1,1+plot_idx)
        nmodes = 4
        scale = svals**2 / (svals**2).sum()
        lines = []
        for i in range(nmodes):
            lns = plt.plot(scale[i] * svecs[i, :, :, 0].T, color = colors[i])
            lines.append(lns[0])

        if plot_idx == 0:
            plt.legend(lines, [f'Mode {i}' for i in range(nmodes)], ncol = 2)

        plt.title('$n_{in}$'+ f' = {cfg.d_input}')

        if plot_idx == 1:
            plt.ylabel('Weighted Temporal Modes')

    plt.xlabel('Time')
    plt.tight_layout()
    plt.savefig('panel_C.pdf')

def panel_D(cfg):
    cfg.use_fourier_features = True
    cfg.d_input = 1
    os.makedirs("data", exist_ok=True)
    plt.figure()
    n_freqs = np.linspace(1, 120, 100).astype(int)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][3:]
    for color, B in zip(colors, [5, 10, 15]):
        ranks_ntk_time, ranks_ntk_space, ranks_v, ranks_ntk = [], [], [], []
        ranks_x = []
        if os.path.exists(f'data/ff_ranks_ntk_time_{B}.npy'):
            ranks_ntk_time = np.load(f'data/ff_ranks_ntk_time_{B}.npy')
            ranks_x = np.load(f'data/ff_ranks_x_{B}.npy')
        else:
            for n_freq in tqdm(n_freqs):
                cfg.fourier_n_freqs = n_freq
                model = SingleBlockAttnMLP(cfg)

                T = 50
                x = torch.randn(B, T, cfg.d_input)

                y, cache = model(x, need_weights=True)

                V_flat = flatten_bt(cache["V"])
                K_core = V_flat @ V_flat.T

                class GetWeightSites(nn.Module):
                    def __init__(self, net):
                        super().__init__()
                        self.net = net

                    def forward(self, x):
                        return self.net(x)[1]["V"] # Relevant quantities.

                from kpflow.grad_op import HiddenNTKOperator as NTK
                model_wrap = GetWeightSites(model)
                vs = model_wrap(x)
                ntk = NTK(model_wrap, x, vs, dev=x.device)
                ntk_time = ntk.partial_avg(-1)

                ranks_ntk_time.append(ntk_time.effrank(nsamp=100, grammian = False))
                ranks_x.append(effdim(cache["x_in"]))

        np.save('data/ff_n_freqs.npy', np.array(n_freqs))
        np.save(f'data/ff_ranks_ntk_time_{B}.npy', np.array(ranks_ntk_time))
        np.save(f'data/ff_ranks_x_{B}.npy', np.array(ranks_x))

        print(n_freqs, ranks_ntk_time)
        plt.plot(n_freqs, ranks_ntk_time, label = f'$n_x = {B}$', color = color)
        plt.plot(n_freqs, ranks_x, label = f'$n_x = {B}$', color = color, linestyle = 'dashed')
        plt.xlabel('Fourier Feature Dimension')
        plt.title('NTK Temporal Rank')

    plt.legend(ncol = 3)
    plt.savefig('panel_D.pdf')
    plt.show()

if __name__ == "__main__":
    set_mpl_defaults(14)
    torch.manual_seed(0)

    cfg = SimpleAttnMLPConfig(
        d_input=8,
        d_model=64,
        n_heads=1,
        mlp_hidden=128,
        mlp_layers=3,   # 64 -> 128 -> 128 -> d_output
        d_output=1,
        max_seq_len=128,
        use_positional_embedding=False,
    )
    panel_D(cfg)
    plt.show()
    panel_C(cfg)
    plt.show()
    panel_B(cfg)



    R = np.random.randn(*svecs[0,:,:,0].shape)
    R = np.random.rand(*svecs[0,:,:,0].shape)
    R = 0*x[:,:,0] + np.mean(x[:, :, 0].detach().numpy(), axis = 1, keepdims = True)
    R = R - np.trace(R.T @ Q) * Q
    R = R / np.linalg.norm(R, ord = 'fro')

    plt.figure()
    plt.subplot(1,3,1)
    print(x.shape)
    plt.plot(x[:, :, 0].T)
    plt.title('Task Input')
    plt.subplot(1,3,2)
    plt.plot(Q.T)
    plt.title('NTK Dominant Mode')
    plt.subplot(1,3,3)
    plt.plot(R.T)
    plt.title('Example Orthogonal Target')
    plt.show()
    asijdoisajd
