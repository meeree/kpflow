import torch, numpy as np, sys, matplotlib.pyplot as plt
from tqdm import tqdm
from kpflow.tasks import CustomTaskWrapper
from kpflow.analysis_utils import ping_dir, load_checkpoints, import_checkpoint
from at_init_fig import construct_model
import glob, os

import sys
sys.path.append('../')
from common import imshow_nonuniform, set_mpl_defaults

set_mpl_defaults(14)
task = CustomTaskWrapper('memory_pro', 30, use_noise = False, n_samples = 30, T = 90)
inputs, targets = task()

V_targ = targets.reshape((-1, targets.shape[-1]))[:, 1:]

path = 'data_lr=0.01/'
files = glob.glob(f'{path}rnn_mempro*/')

# Note that cos(V_1 V_1^T, V_2 V_2^T) = ||V_1^T V_2||_F^2 / (||V_1^T V_1||_F ||V_2^T V_2||_F).

grams = []
Vs = []
grams.append((V_targ @ V_targ.T).detach().numpy())
files = [files[1], files[0]]
Vs.append(V_targ.detach().numpy())
for fname in files[:2]:
    checkpoints, itr = load_checkpoints(fname)
    model = construct_model(g = 1.0, fps = [], dt = .7)
    model.load_state_dict(import_checkpoint(checkpoints[0])['model'])
    out, hidden = model(inputs)
    V = hidden.reshape((-1, hidden.shape[-1]))
    grams.append((V @ V.T).detach().numpy())
    Vs.append(V.detach().numpy())

# Triangle affine hull interpolation between 3 points (Barycentric coords) :)
V_targ, V0, V1 = Vs
V_targ_big = np.zeros_like(V0)
V_targ_big[:, :V_targ.shape[1]] += V_targ
V_targ = V_targ_big

for idx, v in enumerate(Vs):
    h = v.reshape((*hidden.shape[:-1], -1))
    plt.subplot(1,3,1+idx)
    for n in range(h.shape[0]):
        plt.plot(h[n, :, :])

plt.show()

g_targ, g0, g1 = grams
ts = np.linspace(0., 1., 100)
tri_vals = []
x_vals, y_vals = [], []
g_targ_nm = (g_targ * g_targ).sum()**.5
for t0 in tqdm(ts):
    for t1 in ts:
        if t1 + t0 > 1.:
            continue
        t2 = 1 - t0 - t1 # must sum to 1
        V = t0 * V0 + t1 * V_targ + t2 * V1
        G = V.T @ V
        x = 0. * t0 + 1. * t1 + 1. * t2
        y = 0. * t0 + 0. * t1 + 1. * t2
        sim = ((V_targ.T @ V)**2).sum() / (np.linalg.norm(G) * g_targ_nm)
        x_vals.append(x)
        y_vals.append(y)
        tri_vals.append(sim)

x_vals = np.stack(x_vals)
y_vals = np.stack(y_vals)
tri_vals = np.stack(tri_vals)
imshow_nonuniform(x_vals, y_vals, tri_vals)
plt.show()

# ALTERNATIVE
checkpoints = [load_checkpoints(files[0])[0], load_checkpoints(files[1])[0], load_checkpoints(files[1])[-1]]
ntks = []
for ch in checkpoints:
    checkpoints, itr = load_checkpoints(fname)
    checkpoints_all.append(checkpoints)
    model = construct_model(g = 1.0, fps = [], dt = .7)
    model.load_state_dict(import_checkpoint(checkpoints[0])['model'])
    out, hidden = model(inputs)
    V = hidden.reshape((-1, hidden.shape[-1]))
    grams.append((V @ V.T).detach().numpy())
    Vs.append(V.detach().numpy())
    checkpoints_all.append(checkpoints)
