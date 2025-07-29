# 3-bit-flip-flop task
import torch
from torch import nn
import numpy as np

DEFAULT_CFG = {
    'T': 90, 
    'n_samples': 50000,
    'n_spike': 20, 
    'mix_anti': False
}

def generate_poisson_matrix(num_rows, num_cols, m):
    # Calculate the number of ones expected in each column
    num_ones_per_col = np.random.poisson(num_rows / m, num_cols)

    # Initialize the matrix with zeros
    matrix = np.zeros((num_rows, num_cols), dtype=int)

    # Populate the matrix with ones
    for col in range(num_cols):
        # Randomly choose positions for ones in this column
        ones_positions = np.random.choice(num_rows, num_ones_per_col[col], replace=False)
        matrix[ones_positions, col] = np.random.choice([-1, 1], size=num_ones_per_col[col])

    return matrix

def generate(cfg = DEFAULT_CFG, debug = False, noise = True):
    # Inputs are 3 dimensional.
    T = cfg['T']
    mix_anti = cfg['mix_anti']
    D = 3
    
    inp = torch.zeros((cfg["n_samples"], T, D))
    target = torch.zeros_like(inp)
    for d in range(inp.shape[-1]):
        # Generate input.
        n_spike = T / 3 if cfg['n_spike'] == -1 else cfg['n_spike']
        inp[..., d] = torch.from_numpy(generate_poisson_matrix(cfg['n_samples'], T, n_spike))

        # Generate target.
        for i in range(inp.shape[0]):
            keep = 0
            for j in range(inp.shape[1]):
                if inp[i,j,d] != 0: # switch
                    keep = inp[i,j,d]
                target[i,j,d] = keep

    if mix_anti:
        anti_input = torch.zeros_like(inp[:, :, 0:1])
        anti_input += 2 * torch.randint(2, size = (cfg['n_samples'], 1, 1)) - 1
        target = target * anti_input # Negative sign for anti task.
        inp = torch.concatenate((inp, anti_input), -1)

    if debug:
        import matplotlib.pyplot as plt
        plt.figure()

    # TODO vary durations.

    # Add noise to inputs.
    if noise:
        with torch.no_grad():
            inp += torch.normal(torch.zeros_like(inp), 1.) * .03 * (2 ** .5)
    return inp, target

def accuracy(X, Y):
    # Correct if within .5 of goal, since outputs are either [-1, 0, 1].
    return {'acc': torch.mean((torch.abs(X - Y) < .5).float())}
