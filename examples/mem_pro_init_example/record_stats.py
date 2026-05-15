import torch, numpy as np, sys, matplotlib.pyplot as plt
from tqdm import tqdm
from kpflow.tasks import CustomTaskWrapper
from kpflow.propagation_op import PropagationOperator_LinearForm
from kpflow.frechet_op import FrechetOperator 
from kpflow.architecture import BasicRNN, get_cell_from_model
from kpflow.analysis_utils import ping_dir, load_checkpoints, import_checkpoint
from kpflow.grad_op import HiddenNTKOperator as NTK
from kpflow.op_common import IdentityOperator as Id, MatrixWrapper as Mat
from at_init_fig import construct_model
import glob, os
import sys
sys.path.append('../')

from common import project, plot_trajectories, compute_svs, set_mpl_defaults, plot_traj_mempro, imshow_nonuniform, effdim, skree_plot
import argparse

parser = argparse.ArgumentParser(description='Analysis of Recorded Data for Different FP Inits Memory Pro Task')
parser.add_argument('--file', default='data_lr=0.01/', type = str, help='Case to analyze')
args = parser.parse_args()

def get_ntk(model, inputs, hidden):
    class GetHidden(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x)[1]
    return NTK(GetHidden(model), inputs, hidden)

def get_p_and_p_inv(model, inputs, hidden):
    cell = get_cell_from_model(model)
    pop = PropagationOperator_LinearForm(cell, inputs, hidden)
    pop_inv = FrechetOperator(cell, inputs, hidden)
    return pop, pop_inv

task = CustomTaskWrapper('memory_pro', 30, use_noise = False, n_samples = 30, T = 90)
inputs_all, targets_all = task() # For raw attractors.
inputs, targets = inputs_all[:30], targets_all[:30] # Sub-sample for NTK stuff.

targ_np = targets_all.detach().numpy().reshape((-1, targets.shape[-1]))
U, sig, _ = np.linalg.svd(targ_np, full_matrices = False)
U = np.moveaxis(U, -1, 0)
U = U.reshape((U.shape[0], targ_np.shape[0]))

path = args.file
files = glob.glob(f'{path}/*rnn_mempro*/')

pbar = tqdm(files)
for fname in pbar:
    pbar.set_description(fname)

    checkpoints, itr = load_checkpoints(fname)
    checkpoints, itr = checkpoints[::3], itr[::3]
    model = construct_model(g = 1.0, fps = [], dt = .7)

    if not os.path.exists(f'{fname}stats.npz'):
        np.savez(f'{fname}stats.npz', **{})

    stats = dict(np.load(f'{fname}stats.npz'))
    generated = list(stats.keys())
#    generated.remove('rayleigh')
#    generated.remove('targ_proj')
#    generated.remove('attractor')

    keys_all = ['hidden', 'out', 'gd_itrs', 'ntk_modes_init', 'V_modes_init', 'P_modes_init', 'ntk_total_var_init', 'P_total_var_init', 'ntk_svals_init', 'ntk_norms', 'losses', 'rayleigh', 'ntk_effrank', 'targ_proj', 'ntk_v_overlap']
    for name in keys_all:
        if name not in generated:
            stats[name] = [] # Create it.

    for ch in tqdm(checkpoints):
        ld = import_checkpoint(ch)
        model.load_state_dict(ld['model'])

        out, hidden = model(inputs_all)

        if 'out' not in generated:
            stats['out'].append(out.detach().cpu().numpy())

        if 'hidden' not in generated:
            stats['hidden'].append(hidden.detach().cpu().numpy())

        if 'targ_proj' not in generated:
            stats['targ_proj'].append([])
            out_np = out.detach().numpy().reshape((-1, out.shape[-1]))
            print(out_np.shape, U.shape, targ_np.shape)
            for i in range(U.shape[0]):
                # If task solved, so out = targ, out = U Sigma V^T, so <u_i, out> = sigma_i v_i, so ||<u_i, out>|| = sigma_i.
                u_i_out_dot = (out_np * U[i][:,None]).sum(0)
                stats['targ_proj'][-1].append(np.linalg.norm(u_i_out_dot) / sig[i])

        B_sub = inputs.shape[0] # Don't need to use so many samples. Makes it super slow.
        out, hidden = out[:B_sub], hidden[:B_sub]
        ntk = get_ntk(model, inputs, hidden)
        
        class GetHidden(torch.nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, x):
                return self.net(x)[1]

        bipart = lambda x : x.reshape((-1, x.shape[-1])) 
        cell = get_cell_from_model(model)
        V = torch.cat((bipart(hidden), bipart(inputs)), -1)

        if ch == checkpoints[0] and ('ntk_modes_init' not in generated or 'V_modes_init' not in generated or 'P_modes_init' not in generated):
            svals, svecs = ntk.svd(30, (0,1), grammian = False, compute_vecs = True)
            stats['ntk_modes_init'] = svecs
            stats['ntk_svals_init'] = svals

            pop = PropagationOperator_LinearForm(cell, inputs, hidden)
            S = torch.svd(V)[1]
            S_P = pop.svd(30, (0, 1), grammian = True)

            stats['V_modes_init'] = S.detach().numpy()
            stats['P_modes_init'] = S_P

        if ch == checkpoints[0] and ('ntk_total_var_init' not in generated or 'P_total_var_init' not in generated):
            pop = PropagationOperator_LinearForm(cell, inputs, hidden)
            stats['ntk_total_var_init'] = ntk.partial_avg(-1).fro_norm()**2
            stats['P_total_var_init'] = pop.partial_avg(-1).fro_norm()**2

        if 'ntk_v_overlap' not in generated:
            stats['ntk_v_overlap'].append(ntk.partial_avg(-1).alignment(Mat(V @ V.T)))

#        colors = [*plt.rcParams['axes.prop_cycle'].by_key()['color']]
#        plt.plot(np.cumsum((svals ** 2) / (svals**2).sum()))
#        plt.show()
#        for i in range(5):
#            plt.plot(svecs[i, :, :, 0].T * svals[i]**2, color = colors[i])
#        plt.show()
#        print(ntk.effdim((0,1)))
#        asjdsaojd

#        if 'rayleigh' not in generated:
#            targ_sig = targets @ model.Wout.weight.data
#            targ_cpy = torch.zeros_like(targets)
#            targ_cpy += torch.from_numpy(U[2, :, :, None]) # Alginment with mode 3 of target.
#            stats['rayleigh'].append(ntk.rayleigh_coef(targ_cpy @ model.Wout.weight.data))
#
        if 'ntk_norms' not in generated:
            stats['ntk_norms'].append(ntk.fro_norm())

        if 'losses' not in generated:
            stats['losses'].append(torch.nn.MSELoss()(out, targets).item())

        if 'ntk_effrank' not in generated:
            stats['ntk_effrank'].append(ntk.effdim(grammian = False))

    for key, arr in stats.items():
        stats[key] = np.array(arr)

    np.savez(f'{fname}stats.npz', **stats)
