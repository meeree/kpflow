# Analysis of the eigenfunction structure (modes) of operators considered.

from kpflow.tasks import CustomTaskWrapper
from kpflow.analysis_utils import ping_dir, load_checkpoints, import_checkpoint, torch_to_np, np_to_torch, cos_similarity
from kpflow.architecture import Model, get_cell_from_model
from kpflow.parameter_op import ParameterOperator, JThetaOperator
from kpflow.propagation_op import PropagationOperator_DirectForm, PropagationOperator_LinearForm
from kpflow.op_common import AveragedOperator, Operator
from kpflow.trace_estimation import trace_hpp

from common import project, plot_trajectories, compute_svs, set_mpl_defaults

import itertools
from scipy.optimize import curve_fit
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
from sklearn.decomposition import PCA
from tqdm import tqdm

def parse_arguments(parser = None):
    parser = argparse.ArgumentParser(description='Analyze Model on Memory Pro Task') if parser is None else parser
    parser.add_argument('--model', default='gru', type = str, help='Model to use')
    parser.add_argument('--task_str', default = 'memory_pro', type = str, help = 'Task to train on. Options: memory_pro, flip_flop, context_integration')
    return parser.parse_args()

def sine(t, A, f, phi=0.0):
    return np.sin(2 * np.pi * f * (t + phi))  * A

if __name__ == '__main__':
    args = parse_arguments()
    set_mpl_defaults(14)
    PALLETTE = ['#f86969ff', '#7e69f8ff', '#f8c969ff', '#69f87cff', '#e569f8ff']

    task_nice_str = args.task_str.replace('_', ' ').title()
    print(f'Evaluating Results for {task_nice_str}')
    task = CustomTaskWrapper(args.task_str, 200, use_noise = False, n_samples = 200, T = 90)
    inputs, targets = task()
    n_in, n_out = inputs.shape[-1], targets.shape[-1]
    ang = torch.arctan2(inputs[:, 0, 2], inputs[:, 0, 1]).detach().cpu().numpy()

    # Sort inputs based on angle for easier analysis later. 
    new_inds = np.argsort(ang)
    inputs, targets, ang = inputs[new_inds], targets[new_inds], ang[new_inds]

    filename = f'{args.task_str}_{args.model}'
    checkpoints, gd_itr = load_checkpoints(filename)
    print(len(checkpoints), filename)
    print(f'Re-Evaluating {len(checkpoints)} Snapshots in {filename}...')
    test_losses, models, hidden_all = [], [], []
    scales = np.linspace(0., 10., len(checkpoints))
    for idx, ch in enumerate(checkpoints):
        model = Model(input_size = n_in, output_size = n_out, rnn=nn.GRU if args.model == 'gru' else nn.RNN, hidden_size = 256)
        model.load_state_dict(import_checkpoint(ch)['model'])

        # Scale inital weights.
        model.load_state_dict(import_checkpoint(checkpoints[0])['model'])
        for name, param in model.named_parameters():
            if name == 'rnn.weight_hh_l0':
                param.data = param.data * scales[idx]
        model.rnn.flatten_parameters()
        
        out, hidden = model(inputs)
        test_losses.append(nn.MSELoss()(out, targets).item())
        hidden_all.append(torch_to_np(hidden))
        models.append(model)

    hidden_all = np.stack(hidden_all)
    print(f'Hidden shape over all GD snapshots has shape {hidden_all.shape} = (GD Iter, Trial, Time, Hidden Unit)')
    
    # NTK implementations compare. 
    B, T, H = hidden_all[0].shape
    cell = get_cell_from_model(models[0])

    for param in cell.parameters():
        param.requires_grad_()
        
    X = inputs
    hidden = [torch.zeros((X.shape[0], H))]
    hidden[0] = hidden[0].to(X.device)
    for t in range(X.shape[1]):
        hidden.append(cell(X[:, t], hidden[-1]).clone())
    hidden = hidden[1:]
    for h in hidden:
        h.requires_grad_()
        h.retain_grad()

    # Nystrom++ Algorithm.
    def trace_nystpp(A, nsamp):
        d = A.shape[0]
        S = np.random.randint(2, size=(d,nsamp)).astype(float) * 2 - 1 # Either 1 or -1
        G = np.random.randint(2, size=(d,nsamp)).astype(float) * 2 - 1 # Either 1 or -1

        Q, _ = np.linalg.qr(A @ S)
        prod = G - Q @ (Q.T @ G)
        return np.trace((A @ Q).T @ Q) + (1./nsamp) * np.trace((A @ prod).T @ prod)

    # IMP 1: fast.
    # NTK alignment.
    inputs, hidden_all = inputs[[0, 50, 100, 150]], hidden_all[:, [0, 50, 100, 150]]
    cell0 = get_cell_from_model(models[0])
    pop0 = PropagationOperator_LinearForm(cell0, inputs, hidden_all[0])
    kop0 = ParameterOperator(cell0, inputs, hidden_all[0])
#    g0 = (pop0 @ kop0 @ pop0.T())
    g0 = kop0

    cellf = get_cell_from_model(models[-1])
    popf = PropagationOperator_LinearForm(cellf, inputs, hidden_all[-1])
    kopf = ParameterOperator(cellf, inputs, hidden_all[-1])
#    gf = (popf @ kopf @ popf.T())
    gf = kopf

    g0_sp = g0.to_scipy(hidden_all[0].shape)
    gf_sp = gf.to_scipy(hidden_all[0].shape)


    tr = lambda A: trace_hpp(A.to_scipy(hidden_all[0].shape), nsamp = 65)
    tr_cross = tr(g0 @ g0)
    tr = lambda A: trace_nystpp(A.to_scipy(hidden_all[0].shape), nsamp = 65)
    print(tr_cross / np.sqrt(tr(g0 @ g0) * tr(g0 @ g0)))
    jasoidjoisadoi

    nsamps = (10**np.linspace(1, 3, 20)).astype(int)
    trace_guesses = []
    for nsamp in tqdm(nsamps):
        trace_guesses.append(trace_nystpp(g0_sp, nsamp))
    trace_guesses = np.array(trace_guesses)
    plt.plot(nsamps, trace_guesses)

    best = trace_guesses[-1]
    rel_err = np.abs(trace_guesses - best) / np.abs(best)

    plt.figure()
    plt.plot(nsamps, rel_err)
    plt.show()
    jsaoidjsaoid

    # IMP 2: very slow.
    BTK = list(itertools.product(range(B), range(T), range(H)))
    inputs, hidden_all = inputs[[0, 50, 100, 150]], hidden_all[:, [0, 50, 100, 150]]
    for b, t, k in tqdm(BTK):
        grad_btk = list(torch.autograd.grad(hidden[t][b,k], cell.parameters(), retain_graph=True))

    print(df.shape)
    asjdoijsadsaoijd
