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
from common import set_mpl_defaults, imshow_nonuniform

set_mpl_defaults(15)
files = glob.glob('data/sweep/model*.pt')
print('Interpreting the data :).')
plot_idx = []
WIDTHS   = [10, 50, 100, 200]
DEPTHS   = [2, 6, 8, 12]
GAINS    = [0.8, 0.9, 1.0, 1.1]
WIDTHS   += (10**np.linspace(1, np.log10(200), 10)).astype(int).tolist() #[10, 50, 100, 200]
DEPTHS   += np.arange(2, 21).tolist() # [2, 6, 8, 12]
GAINS    += np.linspace(.3, 3, 10).tolist() # [0.8, 0.9, 1.0, 1.1]

lookup = lambda x, arr: np.abs(np.array(arr)-x).argmin()
ra, ka = [],[]
ftles = []

colors = []
vals = []
for fl in tqdm(files):
    ld = torch.load(fl)
    ra.append(float(ld['rep_align']))
    ka.append(float(ld['kern_align']))

    config = ld['config']
    L, N, gain = config['depth'], config['width'], config['gain']
    Lidx, Nidx, gainidx = lookup(L,DEPTHS),lookup(N,WIDTHS),lookup(gain,GAINS)
    colors.append([L / max(DEPTHS), np.log10(N) / np.log10(np.array(WIDTHS)).max(), gain / max(GAINS)])
    colors[-1][1] = N / max(WIDTHS)
    vals.append([L, N, gain])

    ftles.append(ld['ftle'].detach().numpy())
    plot_idx.append(Nidx * 4 + gainidx + 16 * Lidx) # TODO : Make automatic. 

vals = np.array(vals)
ra, ka = np.array(ra), np.array(ka)
colors = np.array(colors)

plt.figure(figsize = (3 * 4, 2 * 3))
for i, (yax, name_y) in enumerate(zip([ra, ka], ['RA', 'KA'])):
    for j, (xax, name_x) in enumerate(zip(vals.T, ['Depth', 'Width', 'Gain'])):
        plt.subplot(2, 3, i * 3 + j + 1)
        color = [0., 0., 0.]
        color[j] = 1.
        plt.scatter(xax, yax, color = color)
        if i == 1:
            plt.xlabel(name_x)
        if j == 0:
            plt.ylabel(name_y)

        coefs = np.polyfit(xax, yax, 1)
        plt.plot(xax, np.poly1d(coefs)(xax), alpha = .5, linewidth = 3, color = 'black', zorder = 10)

plt.tight_layout()
plt.show()

plt.scatter(ra, ka, c = colors)
plt.xlabel('Rep. Alignment (RA)')
plt.ylabel('Kernel\nAlignment (KA)')
plt.show()

plt.figure(figsize = (3 * 4, 4))
for idx in range(4):
    plt.subplot(1,4,1+idx)
    if idx < 3:
        imshow_nonuniform(ra, ka, colors[:, idx])
    else:
        imshow_nonuniform(ra, ka, colors)
    plt.title('Colored by ' + ['Depth', 'Width', 'Gain', 'All (RGB)'][idx])
    if idx == 0:
        plt.xlabel('Rep. Alignment (RA)')
        plt.ylabel('Kernel\nAlignment (KA)')

plt.tight_layout()
plt.savefig('ra_ka.pdf')
plt.show()


plt.figure()

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
