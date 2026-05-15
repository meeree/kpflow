import glob, json, re, argparse, sys
import torch, numpy as np, sys, matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
sys.path.append('../')
from common import project, plot_trajectories, compute_svs, set_mpl_defaults, plot_traj_mempro, imshow_nonuniform, effdim, skree_plot

parser = argparse.ArgumentParser(description='Self-Referential Bias Figure for Paper')
parser.add_argument('--file', default='data_lr=0.01/', type = str, help='Case to analyze')
args = parser.parse_args()

colors = [*plt.rcParams['axes.prop_cycle'].by_key()['color']]
set_mpl_defaults(14)

scale = 0.9
fig1 = plt.figure(figsize = (4*scale, 5*scale))

fname = f'{args.file}/sgd_rnn_mempro_nfps=0_g=1.0/'
stats = dict(np.load(f'{fname}stats.npz'))
print(stats)

hidden, out = stats['hidden'], stats['out'] # (checkpoint, batch, timestep, n_hidden), (checkpoint, batch, timestep, n_out)

plt.plot(stats['ntk_effrank'])
plt.show()

print(effdim(hidden[0], center = False))

print(stats['ntk_effrank'][0])
