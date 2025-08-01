# Simplified MemoryPro task with One Hot Encoding based on Flexible multitask computation in recurrent networks utilizes shared dynamical motifs
import torch
from torch import nn
import numpy as np

DEFAULT_CFG = {
    'T_stim': 30, 
    'T_memory': 30, 
    'T_response': 30, 
    'n_samples': 1000,
    'n_angles': 30,
    'amplitude': 1.,
    'anti': False,
    'delay': False
}

def generate(cfg = DEFAULT_CFG, debug = False, noise = True):
    # Fixation tells us when to output and stim1 is what to copy.
    T = cfg["T_stim"] + cfg["T_memory"] + cfg["T_response"]
    D = 1 + cfg['n_angles']

    memory_start = cfg["T_stim"] 
    response_start = memory_start + cfg["T_memory"]
    
    # Generate data.
    inp = torch.zeros((cfg["n_samples"], T, D))
    target = torch.zeros_like(inp)

    # Fixate until response time. 
    inp[:, :response_start, 0] = 1.
    target[:, :, 0] = inp[:, :, 0] # Same target for fixation.

    buckets = torch.arange(inp.shape[0]) % cfg['n_angles'] # Quantized angles.
    one_hot_ang = nn.functional.one_hot(buckets, cfg['n_angles']).float()[:, None, :] # [n_samples, 1, n_angles].

    if not cfg['delay']:
        # Only input up to memory period.
        inp[:, :memory_start, 1:] = one_hot_ang
    else:
        # Input throughout the task.
        inp[:, :, 1:] = one_hot_ang

    target[:, response_start:, 1:] = one_hot_ang

    if cfg['anti']:
        target[:, :, 1:] = target[:, :, 1:] * -1. # Opposite direction.


    # Add noise to inputs.
    if noise:
        with torch.no_grad():
            inp += torch.normal(torch.zeros_like(inp), 1.) * .1 * (2 ** .5)
    return inp, target

def accuracy(X, Y):
    # Answer is correct if it is within pi/10 and fixation matches.
    cnd1 = torch.sum(torch.abs(X[:, -1, 0] - Y[:, -1, 0]) < 1e-2) # Fixation. UNUSED FOR NOw.
    cnd2 = (torch.abs(X[:, -1, 1] - Y[:, -1, 1]) < np.pi / 10.) # Stim1
    cnd3 = (torch.abs(X[:, -1, 2] - Y[:, -1, 2]) < np.pi / 10.) # Stim2
    return {'acc': torch.mean(torch.logical_and(cnd2, cnd3).float())}
