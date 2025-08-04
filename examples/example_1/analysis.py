from kpflow.tasks import CustomTaskWrapper
from kpflow.analysis_utils import ping_dir, load_checkpoints, import_checkpoint, torch_to_np, np_to_torch, cos_similarity
from kpflow.architecture import Model, get_cell_from_model
from kpflow.operators.parameter_op import ParameterOperator, JThetaOperator
from kpflow.operators.propagation_op import PropagationOperator_DirectForm, PropagationOperator_LinearForm
from kpflow.operators.op_common import AveragedOperator, check_adjoint, Operator

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA

from tqdm import tqdm

def parse_arguments(parser = None):
    parser = argparse.ArgumentParser(description='Analyze Model on Memory Pro Task') if parser is None else parser
    parser.add_argument('--model', default='gru', type = str, help='Model to use')
    parser.add_argument('--task_str', default = 'memory_pro', type = str, help = 'Task to train on. Options: memory_pro, flip_flop, context_integration')
    return parser.parse_args()

def set_mpl_defaults(fontsize = 13): # Fontsize etc
    plt.rc('font', size=fontsize)          # controls default text sizes
    plt.rc('axes', titlesize=fontsize)     # fontsize of the axes title
    plt.rc('axes', labelsize=fontsize)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsize)    # fontsize of the tick labels
    plt.rc('legend', fontsize=fontsize)    # legend fontsize
    plt.rc('figure', titlesize=fontsize)  # fontsize of the figure title
    mpl.rcParams['mathtext.fontset']   = 'cm'        # use Computer Modern
    mpl.rcParams['font.family']        = 'serif'     # make non‑math text serif

if __name__ == '__main__':
    args = parse_arguments()
    set_mpl_defaults(14)
    PALLETTE = ['#f86969ff', '#7e69f8ff', '#f8c969ff', '#69f87cff', '#e569f8ff']

    task_nice_str = args.task_str.replace('_', ' ').title()
    print(f'Evaluating Results for {task_nice_str}')
    task = CustomTaskWrapper(args.task_str, 200, use_noise = False, n_samples = 200, T = 90)
    inputs, targets = task()
    n_in, n_out = inputs.shape[-1], targets.shape[-1]
    ang = np.arctan2(inputs[:, 0, 2], inputs[:, 0, 1])

    # Sort inputs based on angle for easier analysis later. 
    new_inds = np.argsort(ang)
    inputs, targets, ang = inputs[new_inds], targets[new_inds], ang[new_inds]

    filename = f'{args.task_str}_{args.model}'
    checkpoints, gd_itr = load_checkpoints(filename)
    print(len(checkpoints), filename)
    print(f'Re-Evaluating {len(checkpoints)} Snapshots in {filename}...')
    test_losses, models, hidden_all = [], [], []
    for ch in checkpoints:
        model = Model(input_size = n_in, output_size = n_out, rnn=nn.GRU if args.model == 'gru' else nn.RNN, hidden_size = 256)
        model.load_state_dict(import_checkpoint(ch)['model'])
        out, hidden = model(inputs)
        test_losses.append(nn.MSELoss()(out, targets).item())
        hidden_all.append(torch_to_np(hidden))
        models.append(model)
    hidden_all = np.stack(hidden_all)
    print(f'Hidden shape over all GD snapshots has shape {hidden_all.shape} = (GD Iter, Trial, Time, Hidden Unit)')

    plt.figure()
    plt.plot(gd_itr, test_losses)
    plt.title('Test loss')
    plt.xlabel('GD Iteration')
    plt.ylabel('Loss (mse)')

    def project(data):
        data_flat = data.reshape((-1, data.shape[-1]))
        pca = PCA().fit(data_flat) 
        return pca, pca.transform(data_flat).reshape(data.shape)

    print(f'Projecting with PCA...')
    pcas, proj_all, proj_w_out_all = [], [], []
    for hidden, model in zip(hidden_all, models):
        pca, proj = project(hidden)
        pcas.append(pca)
        proj_all.append(proj)

        proj_w_out_all.append(model.Wout.weight.data @ pca.components_.T) # Project the output rows into the pca space.
    proj_all, proj_w_out_all = np.stack(proj_all), np.stack(proj_w_out_all)

    effdims = []
    for pca in pcas:
        effdims.append(1. / np.sum(pca.explained_variance_ratio_**2)) # Participation ratio.

    plt.figure()
    plt.plot(effdims)
    plt.ylabel('PR Dimension')

    def plot_trajectories(data, m = 1, n = 1, i = 1, dim = 3, legend = True, colors = None):
        # data should be shape [batch count, time, hidden dim].
        plt.subplot(m, n, i, projection = None if dim == 2 else '3d')
        for idx, traj in enumerate(data):
            if dim == 3:
                if colors is not None:
                    plt.plot(traj[:, 0], traj[:, 1], traj[:, 2], color = colors[idx])
                else:
                    plt.plot(traj[:31, 0], traj[:31, 1], traj[:31, 2], color = PALLETTE[0])
                    plt.plot(traj[30:61, 0], traj[30:61, 1], traj[30:61, 2], color = PALLETTE[1])
                    plt.plot(traj[60:, 0], traj[60:, 1], traj[60:, 2], color = PALLETTE[2])
            else:
                if colors is not None:
                    plt.plot(traj[:, 0], traj[:, 1], color = colors[idx])
                else:
                    plt.plot(traj[:31, 0], traj[:31, 1], color = PALLETTE[0])
                    plt.plot(traj[30:61, 0], traj[30:61, 1], color = PALLETTE[1])
                    plt.plot(traj[60:, 0], traj[60:, 1], color = PALLETTE[2])
        if legend:
            plt.legend(['stim', 'mem', 'resp'])

    # Plot PCA projection and row space projection
    plt.figure(figsize = (10, 6))
    for idx, proj_data in enumerate([proj_all[0, :], torch_to_np(out)]):
        plot_trajectories(proj_data, 1, 2, 1 + idx)
        if idx == 0:
            axis_colors = ['black', 'brown', 'lavender']
            if proj_w_out_all.shape[1] <= 3:
                for idx, proj_w_out in enumerate(proj_w_out_all[-1]):
                    plt.quiver(0, 0, 0, proj_w_out[0], proj_w_out[1], proj_w_out[2], color = axis_colors[idx], length = 3, normalize = True)

            plt.title('PCA Projection')
            plt.gca().set_xlabel('PC1')
            plt.gca().set_ylabel('PC2')
            plt.gca().set_zlabel('PC3')
        else:
            plt.title('$W_{out}$ Row Space Projection')
            plt.gca().set_xlabel('OUT1')
            plt.gca().set_ylabel('OUT2')
            plt.gca().set_zlabel('OUT3')
    plt.tight_layout()
    
    def compute_svs(op, inp_shape, ncomps, compute_vecs = False, tol = 1e-8):
        op_sp = op.to_scipy(inp_shape, inp_shape, dtype = float, can_matmat = False)
        if compute_vecs:
            singular_vals, singular_vecs = eigsh(op_sp, k = ncomps, return_eigenvectors = True, tol = tol)
            return singular_vals[::-1], singular_vecs[:, ::-1].T.reshape((-1, *inp_shape))

        singular_vals = eigsh(op_sp, k = ncomps, return_eigenvectors = False, tol = tol)
        return singular_vals[::-1]

    print('Beginning analysis with KP-Flow Operators...')

    print('Correlation of Eigenfunctions..."')
    try:
        eigdirs = np.load('eigdirs.npy')
    except:
        eigdirs = []
        for idx, (model, hidden) in enumerate(zip(tqdm(models[:15]), hidden_all[:15])):
            cell = get_cell_from_model(model)
            pop = PropagationOperator_LinearForm(cell, inputs, hidden)
            kop = ParameterOperator(cell, inputs, hidden)
            avg_shape = (1, 1, hidden.shape[2])
            avg_grad_op = AveragedOperator(pop @ kop @ pop.T(), hidden.shape)
            eigdirs.append(compute_svs(avg_grad_op, avg_shape, 3, True, tol = 1e-4)[1][:, 0, 0])

        eigdirs = np.stack(eigdirs)
        np.save('eigdirs.npy', eigdirs)

    sim = np.zeros((eigdirs.shape[0], eigdirs.shape[0]))
    for idx1, eigdir1 in enumerate(eigdirs[:, 0]):
        for idx2, eigdir2 in enumerate(eigdirs[:idx1+1, 0]):
            sim[idx1,idx2] = np.abs(cos_similarity(eigdir1, eigdir2))

    plt.figure()
    plt.imshow(sim, origin = 'lower', extent = [gd_itr[0], gd_itr[-1]]*2)
    plt.xlabel('GD Iteration')
    plt.ylabel('GD Iteration')
    plt.title('$\\delta z$ Cosine Similarity')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'{args.task_str}_corr_changes.pdf')
    plt.show()

    dims = [[], []]
    inputs, hidden_all = inputs[[0, 50, 100, 150]], hidden_all[:, [0, 50, 100, 150]]
    for idx, (model, hidden) in enumerate(zip(tqdm(models[::1]), hidden_all[::1])):
        cell = get_cell_from_model(model)
        pop = PropagationOperator_LinearForm(cell, inputs, hidden)
        kop = ParameterOperator(cell, inputs, hidden)

        avg_shape = (1, hidden.shape[1], 1)
        for idx, op in enumerate([kop, pop]):
            ncomp = 60 if idx == 1 else 20
            avg_gram = AveragedOperator(op @ op.T(), hidden.shape)
            sv = compute_svs(avg_gram, avg_shape, ncomp)
            dims[idx].append(Operator.effrank(sv, .95)[0])

    plt.figure(figsize = (8, 4))
    plt.plot(dims[0], color = PALLETTE[0], linewidth = 3)
    plt.plot(dims[1], color = PALLETTE[1], linewidth = 3)
    plt.legend(['$\\mathcal{K}$', '$\\mathcal{P}$'])
    plt.xlabel('GD Iteration')
    plt.ylabel('Operator Effective Rank')
    plt.title(f'Task = {task_nice_str}')
    plt.tight_layout()
    plt.savefig(f'Effrank_over_gd_{args.task_str}.png')
    plt.show()


    colors = (np.stack([ang, ang * 0., ang * 0.], -1) + np.pi) / (2 * np.pi)
    colors = colors[[0, 50, 100, 150]]
    colors = ['red', 'green', 'blue', 'purple']
    for idx, (model, hidden) in enumerate(zip(tqdm(models[::1]), hidden_all[::1])):
        sv, sfuns = [], []
        for i in range(hidden.shape[0]):
            cell = get_cell_from_model(model)
            jop = JThetaOperator(cell, inputs[i:i+1], hidden[i:i+1]) 
            pop = PropagationOperator_LinearForm(cell, inputs[i:i+1], hidden[i:i+1])
            kop = ParameterOperator(cell, inputs, hidden)
            
            gram_c = pop @ pop.T()   # Controllability Grammian

            avg_shape = (1, hidden.shape[1], 1)
            avg_gram_c = AveragedOperator(gram_c, hidden.shape)

            sv_i, sfuns_i = compute_svs(avg_gram_c, avg_shape, 42, True)

 #           plt.figure()
 #           plt.plot(sv_i)

 #           plt.figure()
 #           dim, varrat = gram_c.effrank(sv_i, .95) 
 #           plt.plot(varrat)
 #           plt.axhline(.95)
 #           plt.title(dim)

 #           plt.show()

            # Re-orient sfuns
            sfuns_i *= np.sign(sfuns_i[:, :, 0:1, :]) 
            sv.append(sv_i); sfuns.append(sfuns_i[:, 0])

        sv = np.stack(sv, 1)
        sfuns = np.stack(sfuns, 1)

        kop = ParameterOperator(cell, inputs, hidden)
        dim, varrat = gram_c.effrank(sv, .95) 

#        plt.figure()
#        plt.plot(sv)
#
#        norm_mean = np.mean(np.linalg.norm(sfuns, axis=-1), -1) # Norm over hidden, mean over time. 
#        plt.figure()
#        plt.plot(norm_mean[0])

        wsfuns = sv[:, :, None, None] * sfuns
        ymin, ymax = wsfuns.min(), wsfuns.max()

        plt.figure(figsize = (12, 8))
        for i in range(wsfuns.shape[0]):
            plt.subplot(6, 7, i+1)
            for j in range(wsfuns.shape[1]):
                plt.plot(wsfuns[i, j, :, :], color = colors[j])
            plt.ylim(ymin, ymax)
        plt.tight_layout()
        
        plt.show()
        plt.savefig(f'anim_frames/anim_{idx}.png')
#
#        plt.figure(figsize = (12, 8))
#        for i in range(20):
#            pca, wsfun_proj = project(wsfuns[i])
#            plot_trajectories(wsfun_proj, 4, 5, i+1, legend = False, dim = 2, colors = colors)
#
#        plt.tight_layout()
#        plt.show()

    plt.show()
