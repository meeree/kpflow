import torch, numpy as np, sys, matplotlib.pyplot as plt
from tqdm import tqdm
from kpflow.tasks import CustomTaskWrapper
from kpflow.architecture import BasicRNN
from kpflow.analysis_utils import ping_dir, load_checkpoints, import_checkpoint
from kpflow.grad_op import HiddenNTKOperator as NTK
from at_init_fig import construct_model
import glob, json, re, argparse
from matplotlib.colors import to_rgba

import sys
sys.path.append('../')
from common import project, plot_trajectories, compute_svs, set_mpl_defaults, plot_traj_mempro, imshow_nonuniform, effdim, skree_plot

parser = argparse.ArgumentParser(description='Analysis of Recorded Data for Different FP Inits Memory Pro Task')
parser.add_argument('--file', default='data_lr=0.01', type = str, help='Case to analyze')
args = parser.parse_args()

colors = [*plt.rcParams['axes.prop_cycle'].by_key()['color']]

path = args.file + '/'
files = glob.glob(f'{path}*rnn_mempro*/')
nfps_all = sorted([int(re.search(r"nfps=(\d+)_g=([0-9]*\.?[0-9]+)", fname).group(1)) for fname in files])

scale = 0.9
fig1 = plt.figure(figsize = (4*scale, 5*scale))
fig2 = plt.figure(figsize = (scale*4, scale*5))
fig3 = plt.figure(figsize = (scale*3*4, scale*3*2))
fig4 = plt.figure(figsize = (4*scale, 5*scale))
pbar = tqdm(files)
set_mpl_defaults(14)
init_cols = ['#ec0d00ff', '#ecaf00ff', '#ff948eff']
for fname in pbar:
    pbar.set_description(fname)
    stats = dict(np.load(f'{fname}stats.npz'))

    m = re.search(r"nfps=(\d+)_g=([0-9]*\.?[0-9]+)", fname)
    nfps, g = int(m.group(1)), float(m.group(2))
    optim = 'Shampoo' if 'shampoo' in fname else 'SGD'
    name = f'{nfps} Fps, {optim}'
    fp_idx = (0 if nfps == 0 else 1)
    idx = fp_idx * 1 + 2 * (1 if optim == 'Shampoo' else 0)

    # Skip this case. Only illustrate Shampoo for bif.
    gd_itrs = stats['gd_itrs']
    if not (nfps != 0 and optim == 'Shampoo'):
        plt.figure(fig1)
        targ_proj = stats['targ_proj']
        cidx = fp_idx + (2 if optim == 'Shampoo' else 0)
        for i in range(targ_proj.shape[1]):
            marker = ['o', '^', 's'][i]
            plt.subplot(3,1,i+1)
            plt.plot(gd_itrs, targ_proj[:,i]**2, color = init_cols[cidx], zorder = -cidx, linewidth = 2)

    #        plt.title(f'Target Mode {i+1}')

            if i == 1:
                plt.ylabel('Target Mode Alignment')
            if i == 2:
                plt.xlabel('GD Iteration')

        losses = stats['losses']
        plt.figure(fig2)
        plt.plot(gd_itrs, losses, color = init_cols[cidx],  linewidth = 3, zorder = -cidx, label = name)
    #    plt.yscale('log')


    plt.figure(fig4)
    rayleigh = stats['rayleigh']
    if not (nfps != 0 and optim == 'Shampoo'):
        plt.plot(gd_itrs, rayleigh, color = init_cols[cidx], label = f'{nfps} FPs', linewidth = 3, zorder = cidx)
        plt.title('NTK Mode 3 Alignment')
#        plt.yscale('log')

    attractor = stats['attractor']
    plt.figure(fig3)
    at0 = attractor[0, :, :, :3]
    atm = attractor[5, :, :, :3]
    atf = attractor[-1, :, :, :3]
    if optim == 'Shampoo':
        cidx = fp_idx 
        for i, at in enumerate([at0, atm, atf]):
            idx = i + 3 * fp_idx
            plt.subplot(2, 3, 1 + idx)
            col = np.array(to_rgba(init_cols[cidx]))[:-1]
            at = at[:, -30:]
            for tidx in range(0, at.shape[1], 5):
                sub = at[:, tidx:tidx+6, :3]
                plt.plot(sub[:,:,1].T, sub[:,:,2].T, color = (.2 + (tidx / at.shape[1])*.8) * col, linewidth = 2)

            if idx == 0:
                plt.title('Before Training')

            if idx == 3:
                plt.xlabel('PC 1')
                plt.ylabel('PC 2')
        
            if idx == 1:
                plt.title('After Training')
                

plt.figure(fig1)
plt.tight_layout()
plt.savefig('targ_mode_alignment.pdf')

plt.figure(fig2)
plt.ylabel('Loss (mse)')
plt.xlabel('GD Iteration')
plt.tight_layout()
plt.legend()
plt.savefig('loses_cases.pdf')

plt.figure(fig3)
plt.tight_layout()
plt.savefig('learned_dyn.pdf')

plt.figure(fig4)
plt.tight_layout()
plt.savefig('mode_ntk_align.pdf')

plt.show()
