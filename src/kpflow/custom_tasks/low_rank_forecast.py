import torch
from torch import nn
import numpy as np

DEFAULT_CFG = {
    'n_samples': 1000,
    'T': 30,
    'dim': 100,
    'latent_dim': 10,
    'seed': 0,
    'seed_data': 1,
    'D_inp': 100,
    'D_targ': 100
}

def generate(cfg = DEFAULT_CFG, debug = False, noise = True):
    T = cfg["T"]
    D = cfg["dim"]
    K = cfg["latent_dim"]

    # Generate data.
    torch.manual_seed(cfg["seed"])
    A = torch.randn(K, K)
    A, _ = torch.linalg.qr(A) # Make A orthogonal.
    C = torch.randn(D, K)

    D_inp = cfg["D_inp"]
    W_inp = torch.randn(D_inp, D) 
    b_inp = torch.randn(D_inp)

    D_targ = cfg["D_targ"]
    W_targ = torch.randn(D_targ, D) 
    b_targ = torch.randn(D_targ)

    if noise:
        z_to_x = lambda  z : z @ C.T + 0.01 * torch.randn(cfg["n_samples"], D)
    else:
        z_to_x = lambda  z : z @ C.T

    torch.manual_seed(cfg["seed_data"])
    z = torch.randn((cfg["n_samples"], K))
    x = torch.zeros((cfg["n_samples"], T+1, D))
    x[:, 0] = z_to_x(z)
    for t in range(T):
        z = 0.8 * z + 0.2 * z @ A.T 
        x[:, t+1] = z_to_x(z)

    inp = x[:, :-1]
    inp = inp @ W_inp.T + b_inp
    inp = torch.sin(inp)

    target = x[:, 1:]  # Predict next
    target = target @ W_targ.T + b_targ
    target = torch.sin(target)

    # Make target high-D
    return inp, target

def accuracy(X, Y):
    return 0.
