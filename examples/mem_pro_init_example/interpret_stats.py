import torch, numpy as np, sys, matplotlib.pyplot as plt
from tqdm import tqdm
from kpflow.tasks import CustomTaskWrapper
from kpflow.architecture import BasicRNN
from kpflow.analysis_utils import ping_dir, load_checkpoints, import_checkpoint
from kpflow.grad_op import HiddenNTKOperator as NTK
from at_init_fig import construct_model
import glob, json, re

colors = [*plt.rcParams['axes.prop_cycle'].by_key()['color']]

path = 'data_lr=0.003/'
files = glob.glob(f'{path}rnn_mempro*/')
nfps_all = sorted([int(re.search(r"nfps=(\d+)_g=([0-9]*\.?[0-9]+)", fname).group(1)) for fname in files])

pbar = tqdm(files)
for fname in pbar:
    pbar.set_description(fname)
    stats = dict(np.load(f'{fname}stats.npz'))

    m = re.search(r"nfps=(\d+)_g=([0-9]*\.?[0-9]+)", fname)
    nfps, g = int(m.group(1)), float(m.group(2))
    color_idx = nfps_all.index(nfps)

    gd_itrs, losses, ntk_norms, pop_min_sing, pop_max_sing, rayleigh = stats['gd_itrs'], stats['losses'], stats['ntk_norms'], stats['pop_min_sing'], stats['pop_max_sing'], stats['rayleigh']

#    plt.plot(gd_itrs, losses, color = colors[color_idx % len(colors)], label = f'{nfps} FPs')
#    plt.plot(gd_itrs, ntk_norms, color = colors[color_idx % len(colors)], label = f'{nfps} FPs')
#    plt.plot(gd_itrs, pop_max_sing / pop_min_sing, color = colors[color_idx % len(colors)], label = f'{nfps} FPs')
    plt.plot(gd_itrs, rayleigh, color = colors[color_idx % len(colors)], label = f'{nfps} FPs')

plt.legend(ncol = 2)
plt.yscale('log')
plt.show()
