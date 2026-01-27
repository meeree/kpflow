import torch, numpy as np, sys, matplotlib.pyplot as plt
from tqdm import tqdm
from kpflow.tasks import CustomTaskWrapper
from kpflow.propagation_op import PropagationOperator_LinearForm
from kpflow.frechet_op import FrechetOperator 
from kpflow.architecture import BasicRNN, get_cell_from_model
from kpflow.analysis_utils import ping_dir, load_checkpoints, import_checkpoint
from kpflow.grad_op import HiddenNTKOperator as NTK
from kpflow.op_common import IdentityOperator as Id
from at_init_fig import construct_model
import glob, os

def get_ntk(model, inputs, hidden):
    class GetHidden(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x)[1]
    return NTK(GetHidden(model), inputs, hidden)

def get_p_and_p_inv(model, inputs, hidden):
    cell = get_cell_from_model(model)
    pop = PropagationOperator_LinearForm(cell, inputs, hidden)
    pop_inv = FrechetOperator(cell, inputs, hidden)
    return pop, pop_inv

task = CustomTaskWrapper('memory_pro', 20, use_noise = False, n_samples = 20, T = 90)
inputs, targets = task()

path = 'data_lr=0.003/'
files = glob.glob(f'{path}rnn_mempro*/')

pbar = tqdm(files)
for fname in pbar:
    pbar.set_description(fname)

    checkpoints, itr = load_checkpoints(fname)
    model = construct_model(g = 1.0, fps = [], dt = .7)

    ntk_norms, losses, pop_max_sing, pop_min_sing, rayleigh = [], [], [], [], []
    cur_stats = dict(np.load(f'{fname}stats.npz'))
    generated = list(cur_stats.keys())
    for ch in checkpoints:
        ld = import_checkpoint(ch)
        model.load_state_dict(ld['model'])

        out, hidden = model(inputs)

        if 'rayleigh' not in generated:
            ntk = get_ntk(model, inputs, hidden)
            raw = ntk.rayleigh_coef(targets @ model.Wout.weight.data)
            rayleigh.append(raw / ntk.op_norm())

        if ('pop_max_sing' not in generated) or ('pop_min_sing' not in generated):
            pop, pop_inv = get_p_and_p_inv(model, inputs, hidden)
            pop_max_sing.append(pop.op_norm())
            pop_min_sing.append(1. / pop_inv.op_norm())

        if 'ntk_norms' not in generated:
            ntk = get_ntk(model, inputs, hidden)
            ntk_norms.append(ntk.fro_norm())

        if 'losses' not in generated:
            losses.append(torch.nn.MSELoss()(out, targets).item())

    stats = {
        'gd_itrs': np.array(itr),
        'ntk_norms': np.array(ntk_norms),
        'losses': np.array(losses),
        'pop_max_sing': np.array(pop_max_sing),
        'pop_min_sing': np.array(pop_min_sing),
        'rayleigh': np.array(rayleigh)
    }

    for name, vals in stats.items():
        if name in generated:
            stats[name] = cur_stats[name]

    np.savez(f'{fname}stats.npz', **stats)
