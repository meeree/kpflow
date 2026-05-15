import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from kpflow.tasks import CustomTaskWrapper
from kpflow.architecture import BasicRNNCell, Model, get_cell_from_model
from tqdm import tqdm
from kpflow.grad_op import HiddenNTKOperator

import sys
sys.path.append('../')
from common import project, plot_trajectories, compute_svs, set_mpl_defaults, plot_traj_mempro, imshow_nonuniform, effdim, relative_error, plot_err_bar, skree_plot

task = CustomTaskWrapper('low_rank_forecast', 20, use_noise = False, n_samples = 20, T = 30, seed_data = 10)#, D_inp = 10, D_targ = 10)
inputs, targets = task()
n_in, n_out = inputs.shape[-1], targets.shape[-1]

def cos(W, V):
    return np.trace(W.T @ V) / (np.linalg.norm(W) * np.linalg.norm(V))

def bias(W):
    return 1 - cos(W, np.eye(W.shape[0]))

set_mpl_defaults(14)
colors = [*plt.rcParams['axes.prop_cycle'].by_key()['color']]

n = 100
W = np.random.randn(n, n) / np.sqrt(n)
noise_vals = [0., 3e-1, 6e-1, 1.]
noise_vals = [0.]
gs = np.linspace(1e-10, 3., 30)
styles = ['solid', 'dashed', 'dotted', '-.']
for idx, (style, noise) in enumerate(zip(styles, noise_vals)):
    cos_vals = []
    dims = []
    effdim_g = []
    effdim_g_time = []
    for g in tqdm(gs):
        Wg = W * g 
        inv = np.linalg.inv(np.eye(n) - Wg)
        inv = inv @ inv.T
        cos_vals.append(bias(inv))

        model = Model(n_in, 256, n_out, rnn = BasicRNNCell, bias = False, linear = True, noise_std = noise)
        model.rnn.weight_hh_l0.data *= g
    #    model.rnn.weight_ih_l0.data *= g
        hidden = model(inputs)[1]

        class GetHidden(torch.nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, x):
                return self.net(x)[1]

        model.rnn.cell.noise = lambda _ : 0. # Disable noise for replay
        gop = HiddenNTKOperator(GetHidden(model), inputs, hidden)
        effdim_g.append(1 - gop.effdim((2,), nsamp = 100, ratio = True, grammian = False))
        effdim_g_time.append(1 - gop.effdim((0,1,), nsamp = 100, ratio = True, grammian = False))

#        cat = torch.cat((hidden, inputs), -1)
#        cat = cat.reshape((-1, cat.shape[-1])).detach().numpy() # time-trials vs space.
#        dims.append(bias(cat @ cat.T))
    #    dims.append(effdim(hidden))

plt.xlabel('$W$ Recurrent Weight Scale, $g$')
plt.legend()
plt.show()
