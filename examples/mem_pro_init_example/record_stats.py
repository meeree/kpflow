import torch, numpy as np, sys, matplotlib.pyplot as plt
from tqdm import tqdm
from kpflow.tasks import CustomTaskWrapper
from kpflow.propagation_op import PropagationOperator_LinearForm
from kpflow.frechet_op import FrechetOperator 
from kpflow.architecture import BasicRNN, get_cell_from_model
from kpflow.analysis_utils import ping_dir, load_checkpoints, import_checkpoint
from kpflow.grad_op import HiddenNTKOperator as NTK
from kpflow.op_common import IdentityOperator as Id
from at_init_fig import construct_model
import glob, os
import sys
sys.path.append('../')

from common import project, plot_trajectories, compute_svs, set_mpl_defaults, plot_traj_mempro, imshow_nonuniform, effdim, skree_plot

parser = argparse.ArgumentParser(description='Analysis of Recorded Data for Different FP Inits Memory Pro Task')
parser.add_argument('--file', default='data_lr=0.01', type = str, help='Case to analyze')
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
inputs, targets = task()

targ_np = targets.detach().numpy().reshape((-1, targets.shape[-1]))
U, sig, _ = np.linalg.svd(targ_np, full_matrices = False)
U = np.moveaxis(U, -1, 0)
U = U.reshape((U.shape[0], *targets.shape[:-1]))

path = args.file
files = glob.glob(f'{path}*rnn_mempro*/')

pbar = tqdm(files)
for fname in pbar:
    pbar.set_description(fname)

    checkpoints, itr = load_checkpoints(fname)
    checkpoints, itr = checkpoints[::3], itr[::3]
    print(len(checkpoints))
    model = construct_model(g = 1.0, fps = [], dt = .7)

    ntk_norms, losses, pop_max_sing, pop_min_sing, rayleigh, ntk_effrank, targ_proj, attractor = [], [], [], [], [], [], [], []
    if not os.path.exists(f'{fname}stats.npz'):
        np.savez(f'{fname}stats.npz', **{})
    cur_stats = dict(np.load(f'{fname}stats.npz'))
    generated = list(cur_stats.keys())
    generated.remove('rayleigh')
#    generated.remove('targ_proj')
    generated.remove('attractor')
    for ch in tqdm(checkpoints):
        ld = import_checkpoint(ch)
        model.load_state_dict(ld['model'])

        out, hidden = model(inputs)

        if 'attractor' not in generated:
            attractor.append(out.detach().cpu().numpy())
#            attractor.append(project(hidden.detach().cpu().numpy())[1])

        if 'rayleigh' not in generated:
            ntk = get_ntk(model, inputs, hidden)
            targ_sig = targets @ model.Wout.weight.data
            targ_cpy = torch.zeros_like(targets)
            targ_cpy += torch.from_numpy(U[2, :, :, None]) # Alginment with mode 3 of target.
            rayleigh.append(ntk.rayleigh_coef(targ_cpy @ model.Wout.weight.data)) 

        if 'targ_proj' not in generated:
            targ_proj.append([])
            out_np = out.detach().numpy().reshape((-1, out.shape[-1]))
            for i in range(U.shape[0]):
                # If task solved, so out = targ, out = U Sigma V^T, so <u_i, out> = sigma_i v_i, so ||<u_i, out>|| = sigma_i.
                u_i_out_dot = (out_np * U[i][:,None]).sum(0)
                targ_proj[-1].append(np.linalg.norm(u_i_out_dot) / sig[i])

#        if ('pop_max_sing' not in generated) or ('pop_min_sing' not in generated):
#            pop, pop_inv = get_p_and_p_inv(model, inputs, hidden)
#            pop_max_sing.append(pop.op_norm())
#            pop_min_sing.append(1. / pop_inv.op_norm())
#
        if 'ntk_norms' not in generated:
            ntk = get_ntk(model, inputs, hidden)
            ntk_norms.append(ntk.fro_norm())

        if 'losses' not in generated:
            losses.append(torch.nn.MSELoss()(out, targets).item())

        if 'ntk_effrank' not in generated:
            ntk_effrank.append(get_ntk(model, inputs, hidden).effdim(nsamp = 40, grammian = False))

    stats = {
        'gd_itrs': np.array(itr),
        'ntk_norms': np.array(ntk_norms),
        'losses': np.array(losses),
        'pop_max_sing': np.array(pop_max_sing),
        'pop_min_sing': np.array(pop_min_sing),
        'rayleigh': np.array(rayleigh),
        'ntk_effrank': np.array(ntk_effrank),
        'targ_proj': np.array(targ_proj),
        'attractor': np.array(attractor)
    }

    for name, vals in stats.items():
        if name in generated:
            stats[name] = cur_stats[name]

    np.savez(f'{fname}stats.npz', **stats)
