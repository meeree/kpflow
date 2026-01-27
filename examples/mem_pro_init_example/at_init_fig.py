from kpflow.tasks import CustomTaskWrapper
from kpflow.analysis_utils import ping_dir, load_checkpoints, import_checkpoint, torch_to_np, np_to_torch, cos_similarity
from kpflow.trace_estimation import trace_hupp_op
from kpflow.architecture import BasicRNN, Model, get_cell_from_model
from kpflow.parameter_op import ParameterOperator, JThetaOperator
from kpflow.propagation_op import PropagationOperator_DirectForm, PropagationOperator_LinearForm
from kpflow.grad_op import HiddenNTKOperator
from kpflow.op_common import Operator, MatrixWrapper, IdentityOperator

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl
import argparse
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import glob, re

import sys
sys.path.append('../')

from common import project, plot_trajectories, compute_svs, set_mpl_defaults, plot_traj_mempro, imshow_nonuniform, effdim, relative_error, plot_err_bar, skree_plot, annotate_subplots

from tqdm import tqdm

def construct_model(g, fps = [], model_dict = None, dt = 1.):
    model = Model(input_size = 3, output_size = 3, rnn = BasicRNN, hidden_size = 256)
    model.rnn.dt = dt
    if model_dict is not None:
        model.load_state_dict(model_dict)
    else:
        with torch.no_grad():
            model.rnn.weight_hh_l0.data *= g
            for dir in fps:
                sig_dir = torch.tanh(dir)
                outer = torch.outer(dir, sig_dir) / torch.linalg.norm(sig_dir)**2
                proj_perp = torch.eye(256) - torch.outer(sig_dir, sig_dir) / torch.linalg.norm(sig_dir)**2 # Orthogonal projector
                W_perp = model.rnn.weight_hh_l0.data @ proj_perp
                model.rnn.weight_hh_l0.data = outer + W_perp
    return model

def eval_model_ode_mode(g, fps = [], ts = 100, dt = 1., h = None, model_dict = None):
    model = construct_model(g, fps, model_dict, dt = dt)
    if h is None:
        h = torch.normal(torch.zeros((inputs.shape[0], 256)), 1.)
    h = torch.zeros_like(h)
    return model, model(inputs, h = h)[1].detach().numpy()

def produce_plot(nfps, gmin = 0., gmax = 2., nruns = 50, twod = False):
    gs = [1.0, 1.5, 1.8, 1.9]
    gs = np.linspace(gmin, gmax, nruns)
    gs = [1.0]
    # gs = [1.8, 1.81]
    torch.random.manual_seed(3)
    cat = []
    fps = [torch.randn(size = (256,)) for _ in range(nfps)]
    for idx, g in enumerate(tqdm(gs)):
        model, activity = eval_model_ode_mode(g, fps = fps, ts = 90, dt = 0.7)#, model_dict = model_dict)
        cat.append(activity)

    cat = np.stack(cat)[0]
    proj = PCA(3).fit_transform(cat.reshape((-1, cat.shape[-1]))).reshape((*cat.shape[:-1],3))
    max_x = np.max(np.abs(proj[:, :, 0]))
    max_y = np.max(np.abs(proj[:, :, 1]))

    class GetHidden(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x)[0]

    bipart = lambda x : x.reshape((-1, x.shape[-1])) # Partition the full 3-tensor space into a time-trials part and a physical part.
    hidden = torch.from_numpy(activity)
#    V = torch.cat((bipart(hidden), bipart(inputs)), -1)
    V = bipart(hidden)
    U,S,_ = torch.svd(V)
    var_rat = S**2 / (S**2).sum()
#    plt.plot(np.cumsum(var_rat), color = colors[len(fps)])
#    plt.subplot(1,5,len(fps)+1)
    lns = []
    for idx in range(5):
        mode = U[:,idx].reshape(hidden.shape[:-1])
        plt_lines = plt.plot((S[idx]**2 * mode.T).detach(), color = colors[idx])
        lns.append(plt_lines[0])
#        plt.plot((S[idx]**2 * mode.T).detach(), color = colors[len(fps)])
    return lns


#   ntk = HiddenNTKOperator(GetHidden(model), inputs, hidden)
#   S_ntk, vecs_ntk = ntk.svd(3, (0, 1), compute_vecs = True)
    
    ax = plt.gca()
    cmap = plt.get_cmap('seismic')
    if twod:
        for i in range(proj.shape[0]):
            ax.plot(proj[i, :, 0], zorder=proj.shape[0]-i)
        plt.xlabel('Time')
        plt.ylabel('PC1')
    else:
        inc = 10
        cmap = plt.get_cmap('hot')
        for off in range(0, proj.shape[1], inc):
            ax.plot(proj[:, off:off+1+inc, 0].T, proj[:, off:off+1+inc, 1].T, proj[:, off:off+1+inc, 2].T, color = cmap(off / proj.shape[1]))
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        ax.zaxis.set_label_text('PC3')
    return None


if __name__ == '__main__':
    colors = [*plt.rcParams['axes.prop_cycle'].by_key()['color']]
    set_mpl_defaults(14)

    task = CustomTaskWrapper('memory_pro', 40, use_noise = False, n_samples = 40, T = 90)
    #task = CustomTaskWrapper('low_rank_forecast', 20, use_noise = False, n_samples = 20, T = 30, seed_data = 10, D_inp = 40, D_targ = 10)
    inputs, targets = task()
    #inputs = torch.cat([inputs] + 3*[torch.zeros_like(inputs)], 1)
    print(inputs.shape)

    plt.figure(figsize = (15, 4))
    for i, nfps in zip(range(4), [0, 1, 2, 5]): 
        plt.subplot(1, 4, i + 1)#, projection = '3d')
        id = f'{nfps+1} FPs'
    #    plt.title(id)
        lns = produce_plot(nfps, nruns = 100)
        if i == 0:
            plt.legend(lns, [f'Mode {idx+1}' for idx in range(5)])
        if i > 2:
            plt.xlabel('Time')
        if i % 3 == 0:
            plt.ylabel('Weighted Mode')

    plt.suptitle('Kronecker Core Modes')
    #plt.suptitle('Dynamics on Memory Pro')
    plt.tight_layout()
    plt.savefig('ntfp_adding.pdf')
    plt.show()
