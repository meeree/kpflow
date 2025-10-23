from kpflow.tasks import CustomTaskWrapper
from kpflow.analysis_utils import ping_dir, load_checkpoints, import_checkpoint, torch_to_np, np_to_torch, cos_similarity
from kpflow.trace_estimation import trace_hupp_op
from kpflow.architecture import BasicRNNCell, Model, get_cell_from_model
from kpflow.parameter_op import ParameterOperator, JThetaOperator
from kpflow.propagation_op import PropagationOperator_DirectForm, PropagationOperator_LinearForm
from kpflow.grad_op import HiddenNTKOperator
from kpflow.op_common import Operator, MatrixOperator, IdentityOperator

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

def effdim_m_mc(Aop, m, n, S=200, rng=None):
    """
    Aop: callable mapping (m,n) -> (m,n)
    Returns (effdim_est, trG_est, trG2_est)
    """
    if rng is None: rng = np.random.default_rng()
    Ys = []
    trG_vals = []
    for _ in range(S):
        X = rng.standard_normal((m, n))
        Y = Aop(X)
        Ys.append(Y)
        trG_vals.append(np.sum(Y*Y))   # ||Y||_F^2
    trG_est = float(np.mean(trG_vals))

    # Unbiased U-statistic for tr(G^2)
    Ys = np.stack(Ys)  # S x m x n
    trG2 = 0.0
    cnt = 0
    for i in range(S):
        for j in range(i+1, S):
            M = Ys[i].T @ Ys[j]
            trG2 += np.sum(M*M)
            cnt += 1
    trG2_est = (2.0 / (S*(S-1))) * trG2 if S > 1 else 0.0

    effdim = (trG_est**2) / trG2_est if trG2_est > 0 else 0.0
    return effdim, trG_est, trG2_est

def predict_rank_K(hidden, inps):
    aug = np.concatenate((hidden, inps), -1)
    return hidden.shape[-1] / effdim(aug, center = False)

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

    model = Model(n_in, 256, n_out, bias = False, rnn = BasicRNNCell) # f(x, h) = W phi(h) + W_in x
#    model = Model(n_in, 256, n_out, bias = False, rnn = nn.RNN) # f(x, h) = W phi(h) + W_in x
    hidden = model(inputs)[1]
    cell = get_cell_from_model(model)
    B, T, H = hidden.shape


    kop = ParameterOperator(cell, inputs, hidden)
    print(cell)

    # Verify kernels perfectly agree. 
    M = np.concatenate((np.tanh(hidden.reshape((B*T, H)).detach().numpy()), inputs.reshape((B*T, n_in)).detach().numpy()), -1)
    K_mat = (M @ M.T)

    validate_k_results = True
    if validate_k_results:
        q = torch.randn(kop.shape_in)
        Q = q.detach().numpy().reshape((B*T, H))
        out = kop(q).detach().numpy()
        guess = (K_mat @ Q).reshape(out.shape)
        print(f'Max norm error guessing K, relative error in outputs {relative_error(guess, out) : .3e}')

        # Estimate ||K||_F^2, squared Frobenius norm:
        true_fro = np.linalg.norm(K_mat, ord='fro') * H**0.5
        guess_fro = trace_hupp_op(kop @ kop.T(), nsamp = 100)**0.5
        print(f'Frobenius norm estimate, relative error: {relative_error(true_fro, guess_fro):.3e}')

        theory_dim = np.linalg.norm(K_mat, ord='fro')**4 / np.linalg.norm(K_mat @ K_mat, ord='fro')**2
        guess_dim = kop.effdim(-1, nsamp = 400)
        print(f"Method 1: H contraction dim, relative error: {relative_error(guess_dim, theory_dim):.3e}, value {theory_dim:.3f}")
    #    kop_stream = lambda x : kop.to_numpy()(x.reshape(kop.shape_in)).reshape(x.shape)
    #    guess_dim = effdim_m_mc(kop_stream, hidden.shape[0]*hidden.shape[1], hidden.shape[2])[0]
    #    print(f"Method 2: H contraction dim, relative error: {relative_error(guess_dim, theory_dim):.3e}")

        theory_dim = H
        guess_dim = kop.effdim((0,1), nsamp = 2000)
        print(f"Method 1: B,T contraction dim, relative error: {relative_error(guess_dim, theory_dim):.3e}, value {theory_dim:.3f}")
    #    kop_stream = lambda x : kop.to_numpy()(x.reshape(kop.shape_in)).reshape(x.shape)
    #    guess_dim = effdim_m_mc(kop_stream, hidden.shape[0]*hidden.shape[1], hidden.shape[2])[0]
    #    print(f"Method 2: B,T contraction dim, relative error: {relative_error(guess_dim, theory_dim):.3e}")

        # Plot the diagonal entries of reduced operator <K K*>_{B,T}, which is an H by H matrix.
        kop_r = kop.partial_trace((0,1)).flatten() # takes in H dimensional inputs.
        plt.plot(np.diag(kop_r.batched_call(torch.eye(H)).detach().numpy()))
        plt.title('Diagonal Entries of <K K*>_{B,T}, Which Should be Constant')
        plt.axhline(np.trace(K_mat))
        plt.show()

    # Now let's look at PK^{1/2}. 
    pop = PropagationOperator_LinearForm(cell, inputs, hidden)
    U, sig2, _ = np.linalg.svd(M, full_matrices = False)
    sig2 = sig2**2
    thin_shape = U.shape[1] # We don't need to give U full (B,T) things!

    U_op = MatrixOperator(U).tprod(IdentityOperator(H))
    qop = pop @ U_op
    Hop = (qop.T() @ qop) # (thin_shape, H) -> (thin_shape, H) operator
    Hop_tr_H = Hop.partial_trace(1).flatten() # (thin_shape,) -> (thin_shape,)
    traces = np.diag(Hop_tr_H.full_matrix()) # a vector of shape thin_shape.
    guess_trace = (sig2 * traces).sum()**2
    
    hutch_trace = (pop @ kop @ pop.T()).trace(nsamp = 500)**2
    hutch_norm = (pop @ kop @ pop.T()).fro_norm(nsamp = 500)**2
    print(relative_error(guess_trace, hutch_trace), hutch_trace)
    print(relative_error(guess_trace, hutch_norm), hutch_norm)
    ajdoisajoid

    guess_dim_direct = (pop @ kop @ pop.T()).effdim((0,1), nsamp = 500, grammian = False)

    print(guess_dim_direct)
