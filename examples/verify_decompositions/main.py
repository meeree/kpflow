from kpflow.tasks import CustomTaskWrapper
from kpflow.analysis_utils import ping_dir, load_checkpoints, import_checkpoint, torch_to_np, np_to_torch, cos_similarity
from kpflow.trace_estimation import trace_hupp_op
from kpflow.architecture import BasicRNNCell, Model, get_cell_from_model
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

bipart = lambda x : x.reshape((-1, x.shape[-1])) # Partition the full 3-tensor space into a time-trials part and a physical part.

def get_operators(D_inp = 10, g = 1., **model_kwargs):
    if args.task_str == 'low_rank_forecast':
        task = CustomTaskWrapper('low_rank_forecast', 20, use_noise = False, n_samples = 20, T = 30, seed_data = 10, D_inp = D_inp, D_targ = 10)
    else:
        task = CustomTaskWrapper(args.task_str, 20, use_noise = False, n_samples = 20, T = 30 if args.task_str != 'memory_pro' else 90)
        
    inputs, targets = task()
    n_in, n_out = inputs.shape[-1], targets.shape[-1]

    model = Model(n_in, 256, n_out, **model_kwargs)
    model.rnn.weight_hh_l0.data *= g # INitial weight scale
    hidden = model(inputs)[1]
    cell = get_cell_from_model(model)

    class GetHidden(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x)[1]

    pop = PropagationOperator_LinearForm(cell, inputs, hidden)
    kop = ParameterOperator(cell, inputs, hidden)
    gop = HiddenNTKOperator(GetHidden(model), inputs, hidden)
    return inputs, targets, hidden, model, cell, pop, kop, gop

def parse_arguments(parser = None):
    parser = argparse.ArgumentParser(description='Analyze Model on Memory Pro Task') if parser is None else parser
    parser.add_argument('--model', default='gru', type = str, help='Model to use')
    parser.add_argument('--task_str', default = 'low_rank_forecast', type = str, help = 'Task to train on. Options: memory_pro, flip_flop, context_integration')
    parser.add_argument('--save_dir', default = '', type = str, help = 'Directory where checkpoints were saved. Optional.')
    return parser.parse_args()

def verify_K_nonlinear_rnn():
    pass

def verify_KP_nonlinear_rnn():
    # Verify that PKP^T = Phi is actually a valid decomposition of the NTK for the hidden state of a nonlinear RNN (in general, any model). 
    pass

def verify_linear_rnn_separable():
    # Verify that NTK is a separable tensor operator for the linear RNN. 
    # Specifically, check NTK = VV^T \otimes ((I_n - W^T) (I_n - W))^{-1} 
    print('''=============================================
Running Verification of Separability for Linear RNN NTK
=============================================='''
    )
    inputs, targets, hidden, model, cell, pop, kop, ntk = get_operators(rnn = BasicRNNCell, bias = False, linear = True) # Linear RNN f(x, h) = W h + W_in x
    inputs, targets, hidden, model, cell, pop, kop, ntk = get_operators(rnn = nn.RNN) # Linear RNN f(x, h) = W h + W_in x
    n_x, n_t, n = hidden.shape
    m = n_x * n_t
    W = model.rnn.weight_hh_l0.data # Weights of the model.

    # Check Jacobians = W.
    if False:
        guess_jacs = torch.zeros_like(pop.jacs)
        guess_jacs += W[None, None]
        print(f"Check Jacobians are all W. Absolute max error: {torch.abs(guess_jacs - pop.jacs).max():.3e}")

        # Check inverse of P.
        L = torch.zeros((n_t, n_t))
        L[range(1, n_t), range(0, n_t-1)] = 1. # Lower diagonal with ones.
        pop_inv_guess = IdentityOperator(n_x*n_t*n).like(pop) - IdentityOperator(n_x).tprod_like(MatrixWrapper(L).tprod(MatrixWrapper(W)), pop)
        composed_l = pop_inv_guess @ pop # Should be identity operator.
        composed_r = pop @ pop_inv_guess
        rel_err_l = composed_l.compare(IdentityOperator(n_x*n_t*n).like(pop), nsamp = 100)
        rel_err_r = composed_r.compare(IdentityOperator(n_x*n_t*n).like(pop), nsamp = 100)
        print(f"Check pop inverse = I - I_{n_x} \otimes L_{n_t} \otimes W, Left Relative Error: {rel_err_l:.3e}, Right Relative Error: {rel_err_r:.3e}")

#        pop_inv_guess = IdentityOperator(n_x*n_t).tprod_like(MatrixWrapper(torch.eye(n)-W), pop)
#        composed = ((pop_inv_guess @ pop) + (pop @ pop_inv_guess)) / 2. # Should be identity operator.
#        rel_err = composed.compare(IdentityOperator(n_x*n_t*n).like(pop), nsamp = 100)
#        print(f"Check pop inverse = I - I_{n_x} \otimes L_{n_t} \otimes W, Relative Error: {rel_err:.3e}")

        # Check full NTK.
        ntk_guess = pop @ kop @ pop.T # This equals the NTK by the KP flow decomposition.
        print(f"Check NTK = P K P.T Composed Operator, Relative Error: {ntk.compare(ntk_guess, nsamp = 40):.3e}")


    S_ntk, vecs_ntk = ntk.svd(20, (0, 1), compute_vecs = True)
    np.save('vecs_ntk.npy', vecs_ntk)
    np.save('S_ntk.npy', S_ntk)

    plt.figure()
    for idx in range(20):
        plt.subplot(4,5,idx+1)
        mode = vecs_ntk[idx].reshape(hidden.shape[:-1])
        plt.plot((S_ntk[idx]**2 * mode.T))
    plt.show()
    
    V = torch.cat((bipart(hidden), bipart(inputs)), -1)
#    V = bipart(hidden)
    U,S,_ = torch.svd(V)

    for idx in range(20):
        plt.subplot(4,5,idx+1)
        mode = U[:,idx].reshape(hidden.shape[:-1])
        plt.plot((S[idx]**2 * mode.T).detach())

    S_ntk = ntk.svd(50, (0, 1))
    S_P = pop.svd(50, (0, 1))

    plt.plot(S.detach()**2)
    plt.figure(figsize = (6, 4))
    rat = np.cumsum((S**2 / (S**2).sum()).detach())
    rat_ntk = np.cumsum((S_ntk**2 / (S_ntk**2).sum()))
    rat_P = np.cumsum((S_P**2 / (S_P**2).sum()))
    rat = np.concatenate([[0,], rat], 0)
    rat_ntk = np.concatenate([[0,], rat_ntk], 0)
    rat_P = np.concatenate([[0,], rat_P], 0)

    plt.plot(range(rat.shape[0]), rat, linewidth = 3)
    plt.plot(range(rat_ntk.shape[0]), rat_ntk, linewidth = 3)
    plt.plot(range(rat_P.shape[0]), rat_P, linewidth = 3)
    plt.axhline(.95, linestyle='dashed', color = 'grey', alpha = 0.5, linewidth = 3)
    plt.text(3, .97, '$\\sigma^2 = .95$')
    plt.legend(['Augmented Activity $V = cat(h, x)$', 'Full Operator $\\text{NTK}_{\\text{cs}} = \mathcal{P K P}^*$', 'Propagation Operator $\mathcal{P}$'])

    plt.xlabel('# of Temporal Modes Used')
    plt.ylabel('Cum. Variance Explained')
    plt.grid()
    annotate_subplots()

    plt.xlim(0, 30)
    plt.tight_layout()
    plt.savefig('./skree_compare.pdf')

    plt.show()

    plt.figure()
    plt.plot(S.detach())
    plt.plot(S_ntk)
    plt.show()
    asjdisaod

    plt.plot(S_ntk)
    plt.show()

    D = torch.linalg.pinv(V.T)
    D_gram = U @ (torch.diag(S**(-2))) @ U.T

    # Time shifted.
    D_ = D.clone().reshape((n_x, n_t, -1))
    D_ = torch.cat((torch.zeros_like(D_[:,:1]), D_), 1)
    D_ = bipart(D_[:, :-1])

    D, D_ = D.reshape((n_x, n_t, -1))[0], D_.reshape((n_x, n_t, -1))[0]
    DD = D - D_

    V_ = V.clone().reshape((n_x, n_t, -1))
    V_ = torch.cat((torch.zeros_like(V_[:,:1]), V_), 1)
    V_ = bipart(V_[:, :-1])
    DV = V - V_

    tau = 0.01

    print(torch.linalg.norm(DV), torch.linalg.norm(DD), tau)
    A0 = DD @ DD.T
    A1 = tau * DD @ D_.T
    A2 = tau**2 * D_ @ D_.T
    plt.subplot(1,3,1)
    plt.imshow(A0.detach())
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(A1.detach())
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.imshow(A2.detach())
    plt.colorbar()
    plt.show()

    print(V.shape)
    plt.subplot(1,2,1)
    plt.plot(U[:,0].reshape((n_x, n_t)).T.detach())
    plt.subplot(1,2,2)
    plt.plot(D.reshape((n_x, n_t, -1))[0, :, ].detach())
    plt.show()


    A0 = (D @ D.T); A1 = (D @ D_.T); A2 = (D_ @ D_.T)

    W_gram = (torch.eye(n) - W.T) @ (torch.eye(n) - W)
    guess_ntk_pinv = (MatrixWrapper(A0).tprod(IdentityOperator(n)) - MatrixWrapper(A1).tprod(MatrixWrapper(W.T)) - MatrixWrapper(A1.T).tprod(MatrixWrapper(W)) + MatrixWrapper(A2).tprod(MatrixWrapper(W@W.T))).like(ntk)
#    guess_ntk_pinv = MatrixWrapper(D_gram).tprod_like(MatrixWrapper(W_gram), ntk)

#    resolvent = torch.linalg.inv(W_gram)
#    guess_ntk = MatrixWrapper(V @ V.T).tprod_like(MatrixWrapper(resolvent), ntk)
#    print(guess_ntk.compare(ntk, nsamp = 50))
#    asdjisajdoi

#    print(ntk.fro_norm(nsamp = 30))
#    print((ntk @ guess_ntk_pinv @ ntk).fro_norm(nsamp = 30))
#    print(guess_ntk_pinv.fro_norm(nsamp = 30))
#    asjdoijsad

    check_1 = (ntk @ guess_ntk_pinv @ ntk).compare(ntk, nsamp = 50)
    print(check_1)
    check_2 = (guess_ntk_pinv @ ntk @ guess_ntk_pinv).compare(guess_ntk_pinv, nsamp = 50)
    print(check_2)
    check_3 = (guess_ntk_pinv @ ntk).T.compare(guess_ntk_pinv @ ntk, nsamp = 50)
    print(check_3)
    check_4 = (ntk @ guess_ntk_pinv).T.compare(ntk @ guess_ntk_pinv, nsamp = 50)
    print(check_4)

    print(f"Check NTK pinverse. Generalized inverse checks: {check_1:.3e}, {check_2:.3e}.")
    print(f"Symmetry of composition checks: {check_3:.3e}, {check_4:.3e}")
    ajsdoisajd



    

#    alignment_guess = guess_ntk.alignment(ntk, nsamp = 100)
#    rel_err = guess_ntk.compare(ntk, nsamp = 100)

    V = torch.cat((bipart(hidden), bipart(inputs)), -1)
    resolvent = torch.linalg.inv((torch.eye(n) - W.T) @ (torch.eye(n) - W))
    ntk = pop @ kop @ pop.T # This equals the NTK by the KP flow decomposition.
    guess_ntk = MatrixWrapper(V @ V.T).tprod_like(MatrixWrapper(resolvent), ntk)
    
    L = torch.zeros((n_t, n_t))
    L[range(1, n_t), range(0, n_t-1)] = 1. # Lower diagonal with ones.

    ntk = pop
    guess_ntk = IdentityOperator(m).tprod_like(MatrixWrapper(torch.linalg.inv(torch.eye(n) - W)), ntk)

    R = torch.zeros((n_t*n, n_t*n))
    plt.figure()
    for d in range(n_t):
        plt.subplot(6, 5, d + 1)
        L = torch.zeros((n_t, n_t))
        L[range(d, n_t), range(0, n_t-d)] = 1
        plt.imshow(L)
        R += torch.kron(torch.matrix_power(W, d), L)
    plt.show()

    guess_ntk = IdentityOperator(n_x).tprod_like(MatrixWrapper(R), ntk)
    alignment_guess = guess_ntk.alignment(ntk, nsamp = 100)
    rel_err = guess_ntk.compare(ntk, nsamp = 100)

    print(f'Operator cosine similarity to guess: 1 + {1-alignment_guess:.3e}')
    print(f'Relative Comparison, operator to guess: {rel_err:.3e}')
    koisadjoisajd

    composed = (guess_ntk @ ntk) # Should be the identity operator.
    print(composed.effdim(nsamp = 100))

#    alignment_guess = guess_ntk.alignment(ntk, nsamp = 100)
#    rel_err = guess_ntk.compare(ntk, nsamp = 100)
    rel_err = composed.alignment(IdentityOperator(n_x*n_t*n).like(composed))

#    print(f'Operator cosine similarity to guess: 1 + {1-alignment_guess:.3e}')
    print(f'Relative Comparison, operator to guess: {rel_err:.3e}')

    print("=============================================")

def vary_scale_input_dim(**model_kwargs):
    if args.task_str == 'low_rank_forecast':
        task = CustomTaskWrapper('low_rank_forecast', 20, use_noise = False, n_samples = 20, T = 30, seed_data = 10, D_inp = D_inp, D_targ = 10)
    else:
        task = CustomTaskWrapper(args.task_str, 20, use_noise = False, n_samples = 20, T = 30 if args.task_str != 'memory_pro' else 90)
        
    inputs, targets = task()
    n_in, n_out = inputs.shape[-1], targets.shape[-1]

    model = Model(n_in, 256, n_out, **model_kwargs)
    model.rnn.weight_hh_l0.data *= g # INitial weight scale
    W = model.rnn.weight_hh_l0.data.detach()
    n = W.shape[0]
    hidden = model(inputs)[1]

    V = torch.cat((bipart(hidden), bipart(inputs)), -1)
    V_gram = V @ V.T
    W_gram = (torch.eye(n) - W.T) @ (torch.eye(n) - W)





if __name__ == '__main__':
    args = parse_arguments()
    set_mpl_defaults(14)
    colors = [*plt.rcParams['axes.prop_cycle'].by_key()['color']]
    verify_linear_rnn_separable()
    oisajdosajd

    task_nice_str = args.task_str.replace('_', ' ').title()
    print(f'Evaluating Results for {task_nice_str}')
    if args.task_str == 'low_rank_forecast':
        task = CustomTaskWrapper('low_rank_forecast', 20, use_noise = False, n_samples = 20, T = 30, seed_data = 10)#, D_inp = 10, D_targ = 10)
    else:
        task = CustomTaskWrapper(args.task_str, 20, use_noise = False, n_samples = 20, T = 30 if args.task_str != 'memory_pro' else 90)
        
    inputs, targets = task()
    n_in, n_out = inputs.shape[-1], targets.shape[-1]

    cell_type = {'gru': nn.GRU, 'rnn': nn.RNN, 'basic_rnn': BasicRNNCell}[args.model]
    model = Model(n_in, 256, n_out, bias = False, rnn = cell_type) # f(x, h) = W phi(h) + W_in x
#    model.rnn.weight_hh_l0.data *= 8
#    model = Model(n_in, 256, n_out, bias = False, rnn = nn.RNN) # f(x, h) = W phi(h) + W_in x
    hidden = model(inputs)[1]
    cell = get_cell_from_model(model)
    B, T, H = hidden.shape
    class GetHidden(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x)[1]

    gop = HiddenNTKOperator(GetHidden(model), inputs, hidden)
    
    if False:
#        wsvecs_all = []
#        ncomp = 5
#        for sidx, scale in enumerate([0, 10]):
#            model = Model(n_in, 256, n_out, bias = False, rnn = cell_type) # f(x, h) = W phi(h) + W_in x
#            model.rnn.weight_hh_l0.data *= scale
#            hidden = model(inputs)[1]
#            cell = get_cell_from_model(model)
#            kop = ParameterOperator(cell, inputs, hidden)
#            svs, svecs = kop.svd(ncomp, keep_dims = (0,1,), compute_vecs = True)
#            svecs = svecs[:, :, :, 0]
#            wsvecs = svs[:,None,None] * svecs
#            wsvecs_all.append(wsvecs)
#
#        wsvecs_all = np.stack(wsvecs_all)
#        ymin, ymax = wsvecs_all.min(), wsvecs_all.max()
#        for sidx in range(2):
#            for i in range(ncomp):
#                plt.subplot(2, ncomp, i + sidx*ncomp+ 1)
#                plt.plot(wsvecs_all[sidx, i].T)
#                plt.ylim(ymin, ymax)
#        plt.show()
#
#        wsvecs_all = []
#        ncomp = 50 
#        for sidx, scale in enumerate([0, 4]):
#            model = Model(n_in, 256, n_out, bias = False, rnn = cell_type) # f(x, h) = W phi(h) + W_in x
#            model.rnn.weight_hh_l0.data *= scale
#            hidden = model(inputs)[1]
#            cell = get_cell_from_model(model)
#            kop = ParameterOperator(cell, inputs, hidden)
#            svs, svecs = kop.svd(ncomp, keep_dims = (1,2), compute_vecs = True)
#            svecs = svecs[:, 0, :, :]
#            wsvecs = svs[:,None,None] * svecs
#            wsvecs_all.append(wsvecs)
#
#        wsvecs_all = np.stack(wsvecs_all)
#        ymin, ymax = wsvecs_all.min(), wsvecs_all.max()
#        for sidx in range(2):
#            plt.subplot(1,2, sidx+1)
#            plt.plot(np.abs(wsvecs_all[sidx]).max(-1).T)
##            for i in range(ncomp):
##                plt.subplot(2, ncomp, i + sidx*ncomp+ 1)
##                plt.plot(wsvecs_all[sidx, i])
##                plt.ylim(ymin, ymax)
#        plt.show()
#
#        for sidx in range(2):
#            proj = project(wsvecs_all[sidx])[1]
#            pca = project(wsvecs_all[sidx])[0]
#            skree_plot(pca, 1, 2, sidx+1)
##            plot_trajectories(proj, 1, 2, sidx + 1)
#        plt.show()

        # Estimate Schmidt rank of operators.
        scales = np.linspace(0., 10., 20) if args.model == 'gru' else np.linspace(0., 3., 20)
        scales = np.array([1.])
        hidden_count = [50, 100, 200, 400, 500, 1000]
        for scale in tqdm(scales):
            schmidt_rank = []
            for n_hid, color in tqdm(zip(hidden_count, colors)):
                model = Model(n_in, n_hid, n_out, bias = False, rnn = cell_type) # f(x, h) = W phi(h) + W_in x
                model.rnn.weight_hh_l0.data *= scale
                hidden = model(inputs)[1]
                cell = get_cell_from_model(model)
                B, T, H = hidden.shape
                kop = ParameterOperator(cell, inputs, hidden)
                pop = PropagationOperator_LinearForm(cell, inputs, hidden)
                gop = HiddenNTKOperator(GetHidden(model), inputs, hidden)

#                a, b = gop.shape_in[0]*gop.shape_in[1], gop.shape_in[2]
#                op_bi = gop.reshape((a, b))
#                nsamp = 30
#                A, B = torch.randn(nsamp,a), torch.randn(nsamp,b)
#                X = A[:, :, None] @ B[:, None, :]
#                Y = op_bi.batched_call(X)
#                var = torch.linalg.svdvals(Y) # [nsamp, hidden count]
#
#                effdim = var.sum(1)**2 / (var**2).sum(1) # [nsamp]
#                schmidt_rank.append(effdim)
                schmidt_rank.append((pop @ kop).alignment(pop @ kop @ pop.T))

            schmidt_rank = np.array(schmidt_rank).T
#            plot_err_bar(hidden_count, schmidt_rank, 25, color = color, label = f'n={n_hid}')
            plt.plot(hidden_count, schmidt_rank)

        plt.xlabel('Random Weight Scale, $g$')
        plt.ylabel('Truncated BPTT Alignment') #effrank$(\mathcal{K}(u v^T))$')
        plt.legend()
        plt.show()

        scales = np.linspace(0., 10., 20) if args.model == 'gru' else np.linspace(0., 3., 20)
        dims = []
        for repeat in range(1):
            dims.append([])
            for scale in tqdm(scales):
                model = Model(n_in, 256, n_out, bias = False, rnn = cell_type) # f(x, h) = W phi(h) + W_in x
                model.rnn.weight_hh_l0.data *= scale
                hidden = model(inputs)[1]
                cell = get_cell_from_model(model)
                B, T, H = hidden.shape
                kop = ParameterOperator(cell, inputs, hidden)
                pop = PropagationOperator_LinearForm(cell, inputs, hidden)
                gop = HiddenNTKOperator(GetHidden(model), inputs, hidden)

                a, b = gop.shape_in[0]*gop.shape_in[1], gop.shape_in[2]
                op_bi = kop.reshape((a, b))
                nsamp = 30
                A, B = torch.randn(nsamp,a), torch.randn(nsamp,b)
                X = A[:, :, None] @ B[:, None, :]
                Y = op_bi.batched_call(X)
                var = torch.linalg.svdvals(Y) # [nsamp, hidden count]

                effdim = var.sum(1)**2 / (var**2).sum(1) # [nsamp]

                rats = np.cumsum((svals**2).sum() **2 / (svals**2).sum(1, keepdims = True), 1)[:, :15]

                plot_err_bar(np.arange(rats.shape[1]), rats, 25, color = [scale / scales.max(), 0., 0.])

                dims[-1].append(kop.effdims([(0,1,2)], ratio = False, nsamp = 80))# + kop.effdims([(0,1), (2,)], ratio = False) + pop.effdims([(0,1), (2,)], ratio = False))
#                dims[-1].append(gop.effdims([(0,1), (2,)], ratio = False, nsamp = 80))# + kop.effdims([(0,1), (2,)], ratio = False) + pop.effdims([(0,1), (2,)], ratio = False))

        plt.show()

        dims = np.stack(dims)
        for i in range(dims.shape[-1]):
            plot_err_bar(scales, dims[...,i], 25, colors[i])
        plt.show()


    kop = ParameterOperator(cell, inputs, hidden)

    # Verify kernels perfectly agree. 
    M = np.concatenate((hidden.reshape((B*T, H)).detach().numpy(), inputs.reshape((B*T, n_in)).detach().numpy()), -1)
    K_mat = (M @ M.T)

    validate_k_results = True
    if validate_k_results:
        kop_guess = MatrixWrapper(K_mat).tprod_like(IdentityOperator(H), kop)
        alignment_guess = kop_guess.alignment(kop, nsamp = 100)
        rel_err = kop.compare(kop_guess, nsamp = 100)

        print(f'Operator cosine similarity to guess: 1 + {1-alignment_guess:.3e}')
        print(f'Relative Comparison, operator to guess: {rel_err:.3e}')

        class GetHidden(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, x):
                return self.net(x)[1]

        gop = HiddenNTKOperator(GetHidden(model), inputs, hidden)
        rel_err = relative_error(effdim(M, center = False), kop.effdim((0,1), nsamp = 300, grammian = False))
        print('effdim([hidden, input]) = effrank_{time,trials}(K) check:'+f' {rel_err:.3e}')

        scales = np.linspace(0., 10., 20) if args.model == 'gru' else np.linspace(0., 3., 20)
        aug_dims, space_dims, time_dims = [], [], []
        for scale in tqdm(scales):
            model = Model(n_in, 256, n_out, bias = False, rnn = cell_type) # f(x, h) = W phi(h) + W_in x
            model.rnn.weight_hh_l0.data *= scale
            hidden = model(inputs)[1]
            cell = get_cell_from_model(model)
            B, T, H = hidden.shape
            kop = ParameterOperator(cell, inputs, hidden)
            gop = HiddenNTKOperator(GetHidden(model), inputs, hidden)
#            pop = PropagationOperator_LinearForm(cell, inputs, hidden)
            op = gop

            M = np.concatenate((hidden.reshape((B*T, H)).detach().numpy(), inputs.reshape((B*T, n_in)).detach().numpy()), -1)
            aug_dims.append(effdim(M, center = False))
            time_dims.append(op.effdim((0,1,), nsamp = 100, grammian = False))
            space_dims.append(op.effdim((2,), nsamp = 100, grammian = False))

        plt.plot(scales, aug_dims)
        plt.plot(scales, space_dims)
        plt.plot(scales, time_dims)
        plt.legend(['(h, x) Activity Dim', 'Operator Hidden Rank', 'Time+Trials Rank'])
        plt.xlabel('Initial Weight Scale, $g$')
        plt.ylabel('Dimensions and Ranks')
        plt.show()


        plt.xlabel('Random Weight Scale, $g$')
        plt.ylabel('Truncated BPTT Alignment') #effrank$(\mathcal{K}(u v^T))$')
        plt.legend()
        plt.show()

#        print(kop.effdims([(0,1,2), (0, 1), (0, 2), (1, 2), (0,), (1,), (2,)], nsamp = 100, grammian = False))
        ajsdoijsad

        # Check K has the form diag(A_1, A_2, ..., A_H)
        q = torch.randn((B, T, H))
        out = kop(q)
        out_stack = []
        for i in range(H):
            qi = torch.zeros_like(q)
            qi[:, :, i] = q[:, :, i]
            out_stack.append(kop(qi)[:, :, i])
        print(f'Check K has form diag_H(A_1, A_2, ..., A_H), relative error: {relative_error(out, torch.stack(out_stack, -1)):.3e}')

        # Check each A_1, A_2, ..., A_H = A is identical
        q = torch.randn((B, T))
        out_stack = []
        for i in range(H):
            qi = torch.zeros((B,T,H))
            qi[:, :, i] = q
            out_stack.append(kop(qi)[:, :, i])
        out_stack = torch.stack(out_stack, -1)
        diffs = (out_stack[:, :, :, None] - out_stack[:, :, None, :]).reshape((-1, H, H)) # [H, H]
        diffs = torch.linalg.norm(diffs, dim=0) / torch.linalg.norm(out_stack.mean(-1).flatten())
        plt.imshow(diffs)
        plt.colorbar()
        plt.show()

        # Estimate ||K||_F^2, squared Frobenius norm:
        true_fro = np.linalg.norm(K_mat, ord='fro') * H**0.5
        guess_fro = trace_hupp_op(kop @ kop.T, nsamp = 100)**0.5
        print(f'Frobenius norm estimate, relative error: {relative_error(true_fro, guess_fro):.3e}')

        theory_dim = np.linalg.norm(K_mat, ord='fro')**4 / np.linalg.norm(K_mat @ K_mat, ord='fro')**2
        guess_dim = kop.effdim((0,1), nsamp = 400)
        print(f"Method 1: B,T dim, relative error: {relative_error(guess_dim, theory_dim):.3e}, value {theory_dim:.3f}")
    #    kop_stream = lambda x : kop.to_numpy()(x.reshape(kop.shape_in)).reshape(x.shape)
    #    guess_dim = effdim_m_mc(kop_stream, hidden.shape[0]*hidden.shape[1], hidden.shape[2])[0]
    #    print(f"Method 2: H contraction dim, relative error: {relative_error(guess_dim, theory_dim):.3e}")

        theory_dim = H
        guess_dim = kop.effdim(-1, nsamp = 2000)
        print(f"Method 1: H dim, relative error: {relative_error(guess_dim, theory_dim):.3e}, value {theory_dim:.3f}")
    #    kop_stream = lambda x : kop.to_numpy()(x.reshape(kop.shape_in)).reshape(x.shape)
    #    guess_dim = effdim_m_mc(kop_stream, hidden.shape[0]*hidden.shape[1], hidden.shape[2])[0]
    #    print(f"Method 2: B,T contraction dim, relative error: {relative_error(guess_dim, theory_dim):.3e}")

        # Plot the diagonal entries of reduced operator <K K*>_{B,T}, which is an H by H matrix.
        kop_r = kop.partial_trace((0,1)).flatten() # takes in H dimensional inputs.
        kop_guess_r = kop_guess.partial_trace((0,1)).flatten()
        plt.plot(np.diag(kop_r.batched_call(torch.eye(H)).detach().numpy()))
        plt.plot(np.diag(kop_guess_r.batched_call(torch.eye(H)).detach().numpy()))
        plt.legend(['diag(tr$_{B,T}(\\mathcal{K}))$', 'diag(tr$_{B,T}(V V^T \\otimes I_H))$'])
        plt.show()


    # Now let's look at PK^{1/2}. 
    pop = PropagationOperator_LinearForm(cell, inputs, hidden)
    M = (M - M.mean(0, keepdims = True))
    U, sig, _ = np.linalg.svd(M, full_matrices = False)
    sig2 = sig**2
    thin_shape = U.shape[1] # We don't need to give U full (B,T) things!

    plt.plot(np.linalg.norm(hidden.detach().numpy(), axis=-1).T)

    plt.figure()

    u1 = U[:, 1]
    plt.plot(u1.reshape(hidden.shape[:-1]).T)
    plt.show()

    U_op = MatrixWrapper(U).tprod(IdentityOperator(H))
    U_op = U_op.reshape(U_op.shape_in, pop.shape_out)
    qop = pop @ U_op 

    check_sigma = (U_op.T @ kop @ U_op)
    randinp = torch.randn(check_sigma.shape_in)
    print(f'Check U.T K U = Sigma^2: relative error {relative_error(torch.mean(check_sigma(randinp) / randinp, 1), sig2):.3e}')

    randinp = torch.randn(pop.shape_in)
    sig2_op = MatrixWrapper(np.diag(sig2)).tprod(IdentityOperator(H))
    pkp_out = (pop @ kop @ pop.T)(randinp)
    qs2q_out = (qop @ sig2_op @ qop.T)(randinp)
    print(f'Check P K P.T = Q (Sigma^2) Q.T : relative error {relative_error(pkp_out, qs2q_out):.3e}')

    sig_op = MatrixWrapper(np.diag(sig)).tprod(IdentityOperator(H))
    Bop = (sig_op @ qop.T).gram() # (shape_thin, H) -> (shape_thin, H)
    Sop = Bop.partial_avg(0).flatten() # (H,) -> (H,)

    plt.plot(np.diag((pop @ kop @ pop.T).partial_avg((0,1)).flatten().full_matrix()))
    plt.plot(np.diag(Sop.full_matrix()))
    plt.show()

#    Sop.set_debug() # Enable class printing at every eval!

#    guess_num = Bop.trace(nsamp = 100)
#    guess_denom = Sop.gram().trace(nsamp = 100)

#    true_num = (pop @ kop @ pop.T).trace(nsamp = 500)
#    true_denom = (pop @ kop @ pop.T).fro_norm(nsamp = 500)**2
    print(relative_error(Bop.effdim(-1, nsamp = 100), (pop @ kop @ pop.T).effdim(-1, nsamp = 100)))
    ajdoisajoid
