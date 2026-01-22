import matplotlib.pyplot as plt
import numpy as np
import torch, os, glob
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tqdm import tqdm
from itertools import product
from train import GetTask, GetModel
from kpflow.grad_op import HiddenNTKOperator as NTK
import sys
sys.path.append('../')
from common import set_mpl_defaults

set_mpl_defaults(14)
files = glob.glob('data/sweep/model*.pt')
print('Interpreting the data :).')
ra, ka = [],[]
for fl in tqdm(files):
    ld = torch.load(fl)
    ra.append(float(ld['rep_align']))
    ka.append(float(ld['kern_align']))

plt.scatter(ra, ka)
plt.xlabel('Representation Alignment (RA)')
plt.ylabel('Kernel Alignment (KA)')
plt.tight_layout()
plt.savefig('ra_ka.png')


plt.figure()
ftles = []
plot_idx = []
WIDTHS   = [10, 50, 100, 200]
DEPTHS   = [2, 6, 8, 12]
GAINS    = [0.8, 0.9, 1.0, 1.1]
lookup = lambda x, arr: arr.index(x)

for fl in tqdm(files):
    ld = torch.load(fl)
    config = ld['config']
    L, N, gain = config['depth'], config['width'], config['gain']
    Lidx, Nidx, gainidx = lookup(L,DEPTHS),lookup(N,WIDTHS),lookup(gain,GAINS)
    ftles.append(ld['ftle'].detach().numpy())
    plot_idx.append(Nidx * 4 + gainidx + 16 * Lidx) # TODO : Make automatic. 

ftles = np.stack(ftles)
nsamp_x = int(ftles.shape[1]**0.5)
ftles = ftles.reshape((-1, nsamp_x, nsamp_x))

plt.figure(figsize = (16 * .9, 4 * 1))
vmin, vmax = ftles.min(), ftles.max()
for i in range(64):
    plt.subplot(4,16,1+plot_idx[i])
    plt.imshow(ftles[i], cmap = 'gist_ncar', origin = 'lower', vmin = vmin, vmax = vmax)
    plt.box(True)
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.savefig('ftles.pdf')
plt.show()
