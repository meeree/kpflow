from kpflow.tasks import CustomTaskWrapper
from kpflow.analysis_utils import ping_dir, load_checkpoints, import_checkpoint, torch_to_np, np_to_torch, cos_similarity
from kpflow.trace_estimation import trace_hupp_op
from kpflow.architecture import BasicRNNCell, Model, get_cell_from_model
from kpflow.parameter_op import ParameterOperator, JThetaOperator
from kpflow.propagation_op import PropagationOperator_DirectForm, PropagationOperator_LinearForm
from kpflow.grad_op import HiddenNTKOperator
from kpflow.op_common import Operator

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

from common import project, plot_trajectories, compute_svs, set_mpl_defaults, plot_traj_mempro, imshow_nonuniform, effdim, relative_error

from tqdm import tqdm

def parse_arguments(parser = None):
    parser = argparse.ArgumentParser(description='Analyze Model on Memory Pro Task') if parser is None else parser
    parser.add_argument('--model', default='gru', type = str, help='Model to use')
    parser.add_argument('--task_str', default = 'memory_pro', type = str, help = 'Task to train on. Options: memory_pro, flip_flop, context_integration')
    parser.add_argument('--save_dir', default = '', type = str, help = 'Directory where checkpoints were saved. Optional.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    set_mpl_defaults(14)
    PALLETTE = ['#f86969ff', '#7e69f8ff', '#f8c969ff', '#69f87cff', '#e569f8ff']

    task_nice_str = args.task_str.replace('_', ' ').title()
    print(f'Evaluating Results for {task_nice_str}')
    if args.task_str == 'low_rank_forecast':
        task = CustomTaskWrapper('low_rank_forecast', 20, use_noise = False, n_samples = 20, T = 30, seed_data = 10, D_inp = 10, D_targ = 10)
    else:
        task = CustomTaskWrapper(args.task_str, 20, use_noise = False, n_samples = 20, T = 30 if args.task_str != 'memory_pro' else 90)
        
    inputs, targets = task()
    n_in, n_out = inputs.shape[-1], targets.shape[-1]

    gs = np.linspace(0, 3, 20)
    effdim_k_BT = []
    effdim_k_H = []
    theory_dims = []
    for cell_type in [nn.RNN]:#[nn.GRU, nn.RNN]:
        effdim_k_BT.append([])
        effdim_k_H.append([])
        effdim_k_H.append([])
        for g in tqdm(gs):
            model = Model(n_in, 256, n_out, bias = False, rnn = cell_type) 
            model.rnn.weight_hh_l0.data *= g

            hidden = model(inputs)[1]
            cell = get_cell_from_model(model)
            kop = ParameterOperator(cell, inputs, hidden)
            pop = PropagationOperator_LinearForm(cell, inputs, hidden)

            # Verify kernels perfectly agree. 
            guess_dim = kop.effdim(-1, nsamp = 100)
            effdim_k_BT[-1].append(guess_dim)
            guess_dim = kop.effdim((0,1), nsamp = 100)
            effdim_k_H[-1].append(guess_dim)
            guess_dim = pop.effdim((0,1), nsamp = 100)
            effdim_k_H[-2].append(guess_dim)

            M = np.concatenate((np.tanh(hidden.reshape((-1, hidden.shape[-1])).detach().numpy()), inputs.reshape((-1, inputs.shape[-1])).detach().numpy()), -1)
            K_mat = (M @ M.T)
            theory_dim = np.linalg.norm(K_mat, ord='fro')**4 / np.linalg.norm(K_mat @ K_mat, ord='fro')**2
            theory_dims.append(theory_dim)

    effdim_k_BT, effdim_k_H = np.array(effdim_k_BT), np.array(effdim_k_H)
    theory_dims = np.array(theory_dims)

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(gs, effdim_k_BT.T)
    plt.plot(gs, theory_dims)
#    plt.legend(['GRU', 'RNN'])
    plt.subplot(1,2,2)
    plt.plot(gs, effdim_k_H.T)
    plt.show()
