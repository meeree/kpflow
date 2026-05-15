import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from kpflow.tasks import CustomTaskWrapper
from kpflow.architecture import BasicRNNCell, Model, get_cell_from_model
from tqdm import tqdm
from kpflow.grad_op import HiddenNTKOperator

import sys, os
sys.path.append('../')
from common import project, plot_trajectories, compute_svs, set_mpl_defaults, plot_traj_mempro, imshow_nonuniform, effdim, relative_error, plot_err_bar, skree_plot, annotate_subplots

resolution = 30

set_mpl_defaults(14)
colors = [*plt.rcParams['axes.prop_cycle'].by_key()['color']]
n = 256
effdim_g_space = []
effdim_g_time = []
model_type = 'rnn'
D_inps = np.linspace(3, 100, resolution).astype(int)
gs = np.linspace(1e-10, 2., resolution) if model_type != 'gru' else np.linspace(1e-10, 9., resolution)
data_dir = f'data/effdims_{model_type}.npy'
print('Data Directory: ', data_dir)
if True or not os.path.exists(data_dir):
    for D_inp in tqdm(D_inps):
        task = CustomTaskWrapper('low_rank_forecast', 20, use_noise = False, n_samples = 20, T = 30, seed_data = 10, D_inp = D_inp, D_targ = 10)
        inputs, targets = task()
        n_in, n_out = inputs.shape[-1], targets.shape[-1]

        W = np.random.randn(n, n) / np.sqrt(n)

        effdim_g_space.append([])
        effdim_g_time.append([])
        for g in gs:
            Wg = W * g 
            inv = np.linalg.inv(np.eye(n) - Wg)
#            inv = inv @ inv.T
            effdim_g_space[-1].append(effdim(inv, center = False))
#            cos_vals.append(bias(inv))

            if model_type == 'gru':
                model = Model(n_in, 256, n_out, rnn = nn.GRU, bias = True)
            elif model_type == 'rnn':
                model = Model(n_in, 100, n_out, rnn = nn.RNN, bias = True)
            else:
                model = Model(n_in, 256, n_out, rnn = BasicRNNCell, bias = False, linear = True)

            model.rnn.weight_hh_l0.data *= g
            hidden = model(inputs)[1]

            joint = torch.cat((hidden, inputs), -1).reshape((-1, inputs.shape[-1]+hidden.shape[-1])).detach().numpy()
            effdim_g_time[-1].append(effdim(joint, center = False))
            continue

            class GetHidden(torch.nn.Module):
                def __init__(self, net):
                    super().__init__()
                    self.net = net

                def forward(self, x):
                    return self.net(x)[1]

            gop = HiddenNTKOperator(GetHidden(model), inputs, hidden)
            effdim_g_space[-1].append(gop.effdim((2,), nsamp = 21, ratio = True, grammian = False))
            effdim_g_time[-1].append(gop.effdim((0,1,), nsamp = 21, ratio = True, grammian = False))

    effdim_g_space, effdim_g_time = np.stack(effdim_g_space), np.stack(effdim_g_time)
    np.save(data_dir, (effdim_g_space, effdim_g_time))

effdim_mine = np.copy(effdim_g_space)
effdim_mine_time = np.copy(effdim_g_time)
effdim_g_space, effdim_g_time = np.load(data_dir)
effdim_g_space *= 256.
effdim_g_time *= 20 * 30.

effdim_g_space = effdim_mine
effdim_g_time = effdim_mine_time

set_mpl_defaults(14)
plt.figure(figsize = (4 * 3, 3))

imshow_fn = lambda x, **kwargs: plt.imshow(x, origin = 'lower', extent = [gs.min(), gs.max(), D_inps.min(), D_inps.max()], **kwargs, aspect = 'auto')

plt.subplot(1,3,1)
imshow_fn(effdim_g_space, cmap = 'viridis')
#plt.xlabel('Recurrent Weight Scale, $g$')
plt.ylabel('Task Input Dim')
plt.title('Spatial Rank')
plt.colorbar()
plt.subplot(1,3,2)
imshow_fn(effdim_g_time, cmap = 'magma')
plt.xlabel('Initial Connectivity Weight Scale, $g$')
plt.title('Temporal Rank')
#plt.colorbar()

plt.subplot(1,3,3)
from scipy.ndimage import gaussian_filter
effdim_g_space = gaussian_filter(effdim_g_space, sigma=1.5)   # try 0.5–3.0
effdim_g_time = gaussian_filter(effdim_g_time, sigma=1.5)   # try 0.5–3.0
mask_1 = effdim_g_space > 30
mask_2 = effdim_g_time > 30
m1t1t = np.logical_and(mask_1, mask_2)
m1t1f = np.logical_and(mask_1, ~mask_2)
m1f1t = np.logical_and(~mask_1, mask_2)
m1f1f = np.logical_and(~mask_1, ~mask_2)
from matplotlib.colors import to_rgba
colors = np.zeros((*effdim_g_space.shape, 4))
colors[m1t1t] = to_rgba('#dd6e42')
colors[m1t1f] = to_rgba('#e8dab2')
colors[m1f1t] = to_rgba('#4f6d7a')
colors[m1f1f] = to_rgba('#1a1423')

imshow_fn(colors)
plt.title('Dynamical Regions')
annotate_subplots()
plt.tight_layout()
#plt.gca().contour(effdim_g_time, levels=[30], colors="k", linewidths=1.5, origin="lower")  # match origin
#plt.gca().contour(effdim_g_space, levels=[30], colors="k", linewidths=1.5, origin="lower")  # match origin
plt.show()
