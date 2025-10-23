from kpflow.tasks import CustomTaskWrapper
from kpflow.analysis_utils import ping_dir, load_checkpoints, import_checkpoint, torch_to_np, np_to_torch, cos_similarity
from kpflow.architecture import Model, get_cell_from_model
from kpflow.parameter_op import ParameterOperator, JThetaOperator
from kpflow.propagation_op import PropagationOperator_DirectForm, PropagationOperator_LinearForm
from kpflow.grad_op import HiddenNTKOperator
from kpflow.op_common import AveragedOperator, Operator

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

from common import project, plot_trajectories, compute_svs, set_mpl_defaults, plot_traj_mempro, imshow_nonuniform, effdim

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

def get_operators(model, inputs, hidden, k=True, p=True, g=True):
    cell = get_cell_from_model(model)
    ops = []
    if k:
        ops.append(ParameterOperator(cell, inputs, hidden))
    if p:
        ops.append(PropagationOperator_LinearForm(cell, inputs, hidden))
    class GetHidden(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x)[1]

    if g:
        ops.append(HiddenNTKOperator(GetHidden(model), inputs, hidden))
    return (*ops,)

def get_models_and_scales(args, n_in, n_out, nscales_default = 10):
    models = []
    if args.save_dir == '':
        scales = np.linspace(0., 10.0, nscales_default) if args.model == 'gru' else np.linspace(0., 3.0, nscales_default)
        for scale in tqdm(scales):
            model = Model(input_size = n_in, output_size = n_out, rnn=nn.GRU if args.model == 'gru' else nn.RNN, hidden_size = 256)
            for name, param in model.named_parameters():
                if name == 'rnn.weight_hh_l0':
                    param.data = param.data * scale
            model.rnn.flatten_parameters()
            models.append(model)
        return models, scales

    # Get final snapshot models from save_dir and infer the g scale from the file names.
    files = glob.glob(f'{args.save_dir}/*/repeat_0/')
    scales = []
    for filename in files:
        checkpoints, gd_itr = load_checkpoints(filename)
        g_val = float(re.search(r"init_([0-9.]+)", filename).group(1)) # Scale should be after init_ in the filename. 
        model = Model(input_size = n_in, output_size = n_out, rnn=nn.GRU if args.model == 'gru' else nn.RNN, hidden_size = 256)
        model.load_state_dict(import_checkpoint(checkpoints[-1])['model'])
        models.append(model)
        scales.append(g_val)
    scales = np.stack(scales)
    inds = np.argsort(scales)
    return [models[idx] for idx in inds], scales[inds]

def power_iteration(op, niter = 30):
    v = torch.randn(*op.shape_in)
    for i in range(niter):
        v /= torch.linalg.norm(v)
        v = op(v)
    return v / torch.linalg.norm(v)

if __name__ == '__main__':
    args = parse_arguments()
    set_mpl_defaults(14)
    PALLETTE = ['#f86969ff', '#7e69f8ff', '#f8c969ff', '#69f87cff', '#e569f8ff']

    task_nice_str = args.task_str.replace('_', ' ').title()
    print(f'Evaluating Results for {task_nice_str}')
    if args.task_str == 'low_rank_forecast':
        task = CustomTaskWrapper(args.task_str, 20, use_noise = False, n_samples = 20, T = 30, seed_data = 10)
    else:
        task = CustomTaskWrapper(args.task_str, 20, use_noise = False, n_samples = 20, T = 30 if args.task_str != 'memory_pro' else 90)
        
    inputs, targets = task()
    n_in, n_out = inputs.shape[-1], targets.shape[-1]

    if args.task_str == 'memory_pro':
        # Sort inputs based on angle for easier analysis later. 
        ang = torch.arctan2(inputs[:, 0, 2], inputs[:, 0, 1]).detach().cpu().numpy()
        new_inds = np.argsort(ang)
        inputs, targets, ang = inputs[new_inds], targets[new_inds], ang[new_inds]

    plot_test_loss = False
    if plot_test_loss:
        test_losses = []
        models, scales = get_models_and_scales(args, n_in, n_out)
        for model in models:
            out = model(inputs)[0]
            loss = nn.MSELoss()(out, targets)
            test_losses.append(loss.item())
        plt.plot(scales, test_losses)
        plt.show()

    # Plot input and target manifold.
    plot_input_manifold = False
    if plot_input_manifold:
        plt.figure()
        plt.subplot(1,1,1,projection= '3d')
        for inp in inputs:
            plt.plot(inp[:, 0], inp[:, 1], inp[:, 2], c = 'black')
        wrap_inputs = np.concatenate([inputs, inputs[:1]], axis = 0)
        plt.plot(*wrap_inputs[:, 0, :].T, color = '#a4ff8aff', linestyle = 'dashed')
        plt.show()

    plot_effranks_pipeline = False
    if plot_effranks_pipeline:
        models, scales = get_models_and_scales(args, n_in, n_out)
        effdims = {'hidden': [], 'adjoint': [], 'input': [], 'target': [], 'delta_f': []}
        for scale, model in zip(tqdm(scales), models):
            hidden, adj, err = model.analysis_mode(inputs, targets)[:3]
            effdims['input'].append(effdim(inputs))
            effdims['target'].append(effdim(targets))
            effdims['hidden'].append(effdim(hidden))
            effdims['adjoint'].append(effdim(adj))

            cell = get_cell_from_model(model)
            pop = PropagationOperator_LinearForm(cell, inputs, hidden)
            kop = ParameterOperator(cell, inputs, hidden)
            delta_f = (kop @ pop.T())(err)
            effdims['delta_f'].append(effdim(delta_f))

        plt.figure()
        plt.plot(scales, effdims['hidden'])
        plt.plot(scales, effdims['adjoint'])
        plt.plot(scales, effdims['delta_f'])
        plt.plot(scales, effdims['input'])
        plt.plot(scales, effdims['target'])
        plt.legend(['Hidden, $h$', 'Adjoint, $a$', 'Delta f, $\delta f$', 'Input, $x$', 'Target, $y^*$'])
        plt.ylabel('Effective Dimension')
        plt.xlabel('Initial Weight Scale, $g$')
        plt.show()

    plot_dim_ratios_phi = True
    if plot_dim_ratios_phi:
        scale = 8
        if True:
            D_inps = np.linspace(1, 500, 20).astype(int)
            D_targs = np.linspace(1, 500, 20).astype(int)
            D_inp_targ = np.array(np.meshgrid(D_inps, D_targs))
            true_D_inp_targ, dim_dh, dim_err, dim_h, dim_adj = [], [], [], [], []
            dim_k, dim_p, dim_phi = [], [], []
            for (D_inp, D_targ) in tqdm(D_inp_targ.reshape(2, -1).T):
                task = CustomTaskWrapper('low_rank_forecast', 20, use_noise = False, n_samples = 20, T = 30, seed_data = 10, D_inp = D_inp, D_targ = D_targ)
                sweep_inputs, sweep_targets = task()
                model = Model(input_size = sweep_inputs.shape[-1], output_size = sweep_targets.shape[-1], rnn=nn.GRU if args.model == 'gru' else nn.RNN, hidden_size = 256)
                for name, param in model.named_parameters():
                    if name == 'rnn.weight_hh_l0':
                        param.data = param.data * scale
                model.rnn.flatten_parameters()
                hidden, adj, err = model.analysis_mode(sweep_inputs, sweep_targets)[:3]
                gop, = get_operators(model, sweep_inputs, hidden, p=False, k=False)

                kop, pop, gop = get_operators(model, sweep_inputs, hidden)
                reduce_shape = (1, 1, hidden.shape[-1])
                reduce_shape = (*hidden.shape[:-1], 1)

                kop_stream = lambda x : kop.to_numpy()(x.reshape(kop.shape_in)).reshape(x.shape)
                print(effdim_m_mc(kop_stream, hidden.shape[0]*hidden.shape[1], hidden.shape[2]))
                print(kop.effdim(reduce_shape, nsamp = 100), predict_rank_K(hidden.detach().numpy(), sweep_inputs.detach().numpy()))
                jdsaooisada

#                s1, s2 = [], []
#                for S in np.linspace(100, 3000, 5).astype(int):
#                    rand_inps = torch.randn((S, *inputs.shape[:-1], 256)) * 1e-6
#                    s1.append(effdim(kop.batched_call(rand_inps)))
#                    s2.append(kop.effdim(reduce_shape, nsamp = S))
#                plt.plot(s1)
#                plt.plot(s2)
#                plt.show()
#                saoijdsadj
#                dim_k.append(effdim(kop.batched_call(rand_inps)))
#                dim_p.append(effdim(pop.batched_call(rand_inps)))
#                dim_phi.append(effdim(gop.batched_call(rand_inps)))
                dim_k.append(kop.effdim(reduce_shape, nsamp = 5))
                dim_p.append(pop.effdim(reduce_shape, nsamp = 5))
                dim_phi.append(gop.effdim(reduce_shape, nsamp = 5))

                dim_h.append(effdim(hidden))
                dim_adj.append(effdim(adj))
                dim_dh.append(effdim(gop(err)))
                dim_err.append(effdim(err))
                true_D_inp_targ.append([effdim(sweep_inputs), effdim(sweep_targets)])

            dim_h = np.stack(dim_h).reshape(D_inp_targ.shape[1:])
            dim_adj = np.stack(dim_adj).reshape(D_inp_targ.shape[1:])
            dim_dh = np.stack(dim_dh).reshape(D_inp_targ.shape[1:])
            dim_err = np.stack(dim_err).reshape(D_inp_targ.shape[1:])

            dim_k = np.stack(dim_k).reshape(D_inp_targ.shape[1:])
            dim_p = np.stack(dim_p).reshape(D_inp_targ.shape[1:])
            dim_phi = np.stack(dim_phi).reshape(D_inp_targ.shape[1:])

            true_D_inp_targ = np.stack(true_D_inp_targ).T.reshape(D_inp_targ.shape)

            ping_dir('data/')
            np.save(f'data/phi_true_D_inp_targ_{scale}.npy', true_D_inp_targ)

            np.save(f'data/phi_dim_k_{scale}.npy', dim_k)
            np.save(f'data/phi_dim_p_{scale}.npy', dim_p)
            np.save(f'data/phi_dim_phi_{scale}.npy', dim_phi)

            np.save(f'data/phi_dim_h_{scale}.npy', dim_h)
            np.save(f'data/phi_dim_adj_{scale}.npy', dim_adj)
            np.save(f'data/phi_dim_dh_{scale}.npy', dim_dh)
            np.save(f'data/phi_dim_err_{scale}.npy', dim_err)

        true_D_inp_targ = np.load(f'data/phi_true_D_inp_targ_{scale}.npy')
        dim_dh, dim_err = np.load(f'data/phi_dim_dh_{scale}.npy'), np.load(f'data/phi_dim_err_{scale}.npy')
        dim_h, dim_adj = np.load(f'data/phi_dim_h_{scale}.npy'), np.load(f'data/phi_dim_adj_{scale}.npy')
        dim_k, dim_p, dim_phi = np.load(f'data/phi_dim_k_{scale}.npy'), np.load(f'data/phi_dim_p_{scale}.npy'), np.load(f'data/phi_dim_phi_{scale}.npy')
#        true_D_inp_targ = np.load(f'data/phi_true_D_inp_targ.npy')
#        dim_dh, dim_err = np.load(f'data/phi_dim_dh.npy'), np.load(f'data/phi_dim_err.npy')
#        dim_h, dim_adj = np.load(f'data/phi_dim_h.npy'), np.load(f'data/phi_dim_adj.npy')
#        dim_k, dim_p, dim_phi = np.load(f'data/phi_dim_k.npy'), np.load(f'data/phi_dim_p.npy'), np.load(f'data/phi_dim_phi.npy')
        ratios = dim_dh / dim_err

        X, Y = true_D_inp_targ
        name_X, name_Y = 'x', 'y*'

        plt.figure(figsize = (12, 3))
        vmin = min(np.min(dim_k), np.min(dim_p), np.min(dim_phi))
        vmax = max(np.max(dim_k), np.max(dim_p), np.max(dim_phi))
        for idx, (dims, name, cmap) in enumerate(zip([dim_k, dim_p, dim_phi], ['K', 'P', '$\\Phi$'], ['viridis', 'plasma', 'coolwarm'])):
            plt.subplot(1,3,1+idx)
            imshow_nonuniform(X, Y, dims, cmap = cmap, aspect = 'auto')#, vmin = vmin, vmax = vmax)
            plt.title(f'effdim({name})')
            plt.xlabel(f'effdim({name_X})')
            plt.ylabel(f'effdim({name_Y})')
            plt.colorbar()
        plt.tight_layout()

        plt.figure(figsize = (8, 3))
        plt.subplot(1,2,1)
        imshow_nonuniform(*true_D_inp_targ, dim_h, cmap = 'hot', aspect = 'auto')
        plt.title(f'effdim(hidden)')
        plt.xlabel('effdim(x)')
        plt.ylabel('effdim(y*)')
        plt.colorbar()
        plt.subplot(1,2,2)
        imshow_nonuniform(*true_D_inp_targ, dim_adj, cmap = 'hot', aspect = 'auto')
        plt.title(f'effdim(adjoint)')
        plt.xlabel('effdim(x)')
        plt.ylabel('effdim(y*)')
        plt.colorbar()
        plt.tight_layout()

        plt.figure()
        imshow_nonuniform(*true_D_inp_targ, dim_dh, cmap = 'coolwarm', aspect = 'auto')
        plt.title(f'effdim($\\delta h$)')
        plt.xlabel('effdim(x)')
        plt.ylabel('effdim(y*)')
        plt.colorbar()
        plt.tight_layout()

        plt.figure()
        imshow_nonuniform(dim_h, dim_err, dim_dh, cmap = 'coolwarm', aspect = 'auto')
        plt.title(f'effdim($\\delta h$)')
        plt.xlabel('effdim(hidden)')
        plt.ylabel('effdim(Err)')
        plt.colorbar()
        plt.tight_layout()
        plt.show()

        plt.figure()
        imshow_nonuniform(*true_D_inp_targ, ratios, cmap = 'seismic', aspect = 'auto', vmin = 0.1, vmax= 1.9)
        #imshow_nonuniform(*true_D_inp_targ, ratios, cmap = 'tab20', aspect = 'auto')#, vmin = 0.1, vmax= 1.9)
        plt.xlabel('effdim(x)')
        plt.ylabel('effdim(y*)')
        plt.title('effdim($\\delta h$)/effdim(Err)')
        plt.colorbar()
        plt.tight_layout()
        plt.show()


    # Measure dominant mode of operators.
    plot_op_dom_mode = False
    if plot_op_dom_mode:
        models, scales = get_models_and_scales(args, n_in, n_out)
        modes = {'k': []}
        sim = {'k': []}
        for scale, model in zip(tqdm(scales), models):
            hidden = model(inputs)[1]
            cell = get_cell_from_model(model)
            kop = ParameterOperator(cell, inputs, hidden)
            modes['k'].append(power_iteration(kop))
            sim['k'].append(cos_similarity(hidden, modes['k'][-1]).item())

            plt.figure(figsize = (8, 3))
            plot_fn = plot_trajectories if args.task_str != 'memory_pro'else (lambda *args : plot_traj_mempro(ang, *args))
            plot_fn(project(hidden.detach().numpy())[1], 1, 2, 1)
            plot_fn(project(modes['k'][-1].detach().numpy())[1], 1, 2, 2)
            plt.show()

        for idx, (name, sim_op) in enumerate(sim.items()):
            plt.plot(sim_op)
        plt.show()
        
        for idx, (name, modes_op) in enumerate(modes.items()):
            plt.figure()
            for idx2, mode in enumerate(modes_op):
#                plt.subplot(2, len(scales)//2, 1 + idx2, projection = '3d')
                plt.subplot(1, len(scales), 1 + idx2, projection = '3d')
                proj = project(mode)[1]
                for b in range(proj.shape[0]):
                    plt.plot(proj[b, :, 0], proj[b, :, 1], proj[b, :, 2])
                plt.xlabel('PC1')
                plt.ylabel('PC2')

        plt.show()

    # Measure amplification of rank by operators for random inputs.
    plot_effrank_amp = True
    if plot_effrank_amp:
        models, scales = get_models_and_scales(args, n_in, n_out)
        rand_inps = torch.randn((10, *inputs.shape[:-1], 256)) * 10
        effdims = {'k': [], 'phi': [], 'p': [], 'state': []} 
        for scale, model in zip(tqdm(scales), models):
            hidden = model(inputs)[1]
            kop, pop, gop = get_operators(model, inputs, hidden)
            effdims['state'].append([effdim(hidden)])
            effdims['k'].append([effdim(out) for out in kop.batched_call(rand_inps)])
            effdims['phi'].append([effdim(out) for out in gop.batched_call(rand_inps)])
            effdims['p'].append([effdim(out) for out in pop.batched_call(rand_inps)])
        
        for idx, (name, efd) in enumerate(effdims.items()):
    #        if name == 'p':
    #            continue
            efd = np.array(efd)
            mn, sd = np.mean(efd, 1), np.std(efd, 1)
            plt.plot(scales, mn, label = name)
            plt.fill_between(scales, mn - 2*sd, mn + 2*sd, alpha = 0.3, label = '_nolegend_')

        plt.legend()
        plt.show()

    # Measure output of operators for random inputs.
    plot_rand_outputs = True
    if plot_rand_outputs:
        models, scales = get_models_and_scales(args, n_in, n_out)
        rand_inp = torch.randn((1, *inputs.shape[:-1], 256))
        outs = {'k': [], 'phi': [], 'p': []} 
        for scale, model in zip(tqdm(scales), models):
            hidden = model(inputs)[1]
            kop, pop, gop = get_operators(model, inputs, hidden)
            outs['p'].append(torch_to_np(pop.batched_call(rand_inp))[0])
            outs['k'].append(torch_to_np(kop.batched_call(rand_inp))[0])
            outs['phi'].append(torch_to_np(gop.batched_call(rand_inp))[0])

        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for idx1, (name, out_over_scale) in enumerate(outs.items()):
            for idx2, (scale, out) in enumerate(zip(scales, out_over_scale)):
                plt.subplot(3, scales.shape[0], idx2 + idx1 * scales.shape[0] + 1)
                if idx1 == 2:
                    plt.xlabel(f'Scale = {scale:.2f}')
                if idx2 == 0:
                    plt.ylabel(f'{name} Operator')

                proj = project(out)[1]
                for b in range(proj.shape[0]):
                    plt.plot(*proj[b, :, :2].T, color = default_colors[idx1])

        plt.tight_layout()
        plt.show()

    plt.show()
