# Simplified MemoryPro task based on Flexible multitask computation in recurrent networks utilizes shared dynamical motifs
import torch
from torch import nn
import numpy as np

DEFAULT_CFG = {
    'T_stim': 30, 
    'T_memory': 30, 
    'T_response': 30, 
    'n_samples': 1000,
    'amplitude': 1.,
    'anti': False,
    'delay': False
}

def generate(cfg = DEFAULT_CFG, debug = False, noise = True):
    # Inputs are 3 dimensional of format,
    # [fixation, stim1 cos, stim1 sin].
    # Fixation tells us when to output and stim1 is what to copy.
    T = cfg["T_stim"] + cfg["T_memory"] + cfg["T_response"]
    D = 3

    memory_start = cfg["T_stim"] 
    response_start = memory_start + cfg["T_memory"]
    
    # Generate data.
    inp = torch.zeros((cfg["n_samples"], T, D))
    target = torch.zeros_like(inp)

    # Fixate until response time. 
    inp[:, :response_start, 0] = 1.
    target[:, :, 0] = inp[:, :, 0] # Same target for fixation.

    # Uniform angles from 0 to 2 * pi.
    angles = torch.rand(inp.shape[0]) * 2 * np.pi
    c, s = cfg["amplitude"] * torch.cos(angles), cfg["amplitude"] * torch.sin(angles)

    if not cfg['delay']:
        # Only input up to memory period.
        inp[:, :memory_start, 1] = c.reshape((-1, 1))
        inp[:, :memory_start, 2] = s.reshape((-1, 1))
    else:
        # Input throughout the task.
        inp[:, :, 1] = c.reshape((-1, 1))
        inp[:, :, 2] = s.reshape((-1, 1))

    target[:, response_start:, 1] = c.reshape((-1, 1))
    target[:, response_start:, 2] = s.reshape((-1, 1))

    if cfg['anti']:
        target[:, :, 1:] = target[:, :, 1:] * -1. # Opposite direction.

    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        for chan, name in enumerate(['Fixation', 'Stim1 Cos', 'Stim1 Sin']):
            for qidx, quant in enumerate([inp, target]):
                plt.subplot(3, 2, 1 + 2*chan + qidx)
                plt.plot(quant[0, :, chan], linewidth = 4)
                plt.title(name)
                plt.axvline(memory_start, c = 'black', alpha = .5, linewidth = 3, linestyle = 'dashed')
                plt.axvline(response_start, c = 'black', alpha = .5, linewidth = 3, linestyle = 'dashed')
                plt.ylim(-1.2, 1.2)
            plt.suptitle('Input Left, Target Right')


    # TODO vary durations.

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
