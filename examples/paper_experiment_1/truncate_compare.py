# Analysis of the eigenfunction structure (modes) of operators considered.

from kpflow.tasks import CustomTaskWrapper
from kpflow.analysis_utils import ping_dir, load_checkpoints, import_checkpoint, torch_to_np, np_to_torch, cos_similarity
from kpflow.architecture import Model, get_cell_from_model
from kpflow.parameter_op import ParameterOperator, JThetaOperator
from kpflow.propagation_op import PropagationOperator_DirectForm, PropagationOperator_LinearForm
from kpflow.grad_op import HiddenNTKOperator
from kpflow.op_common import AveragedOperator, Operator
from kpflow.trace_estimation import trace_hupp

from common import project, plot_trajectories, compute_svs, set_mpl_defaults

import matplotlib.gridspec as gridspec
import torch, re
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
    parser.add_argument('--files', nargs = '+', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    set_mpl_defaults(14)
    PALLETTE = ['#f86969ff', '#7e69f8ff', '#f8c969ff', '#69f87cff', '#e569f8ff']

    task_nice_str = args.task_str.replace('_', ' ').title()
    print(f'Evaluating Results for {task_nice_str}')
    task = CustomTaskWrapper(args.task_str, 50, use_noise = False, n_samples = 50, T = 90) # Uniform inputs without randomization.
    inputs, targets = task()
    n_in, n_out = inputs.shape[-1], targets.shape[-1]
    ang = torch.arctan2(inputs[:, 0, 2], inputs[:, 0, 1]).detach().cpu().numpy()

    checkpoints = args.files
    print(checkpoints)

    test_losses, models, hidden_all = [], [], []
    for idx, ch in enumerate(checkpoints):
        model = Model(input_size = n_in, output_size = n_out, rnn=nn.GRU if args.model == 'gru' else nn.RNN, hidden_size = 256)
        model.load_state_dict(import_checkpoint(ch)['model'])
        
        out, hidden = model(inputs)
        test_losses.append(nn.MSELoss()(out, targets).item())
        hidden_all.append(torch_to_np(hidden))
        models.append(model)
    hidden_all = np.stack(hidden_all)
    print(f'Hidden shape over all GD snapshots has shape {hidden_all.shape} = (GD Iter, Trial, Time, Hidden Unit)')

    fig = plt.figure(figsize = (4 * 5, 3 * len(checkpoints)))
    gs = gridspec.GridSpec(len(checkpoints), 5, figure=fig)

    g_vals = np.load('minimal_rnn_mempro/g_vals.npy')
    alignment_all = np.load('minimal_rnn_mempro/alignment_all.npy')
    first_col = fig.add_subplot(gs[:, 0])
    plt.plot(alignment_all, g_vals)
    plt.title('KA(P K, P K P*)')
    plt.ylabel('g scale')

    for idx, (model, hidden) in enumerate(zip(tqdm(models[::1]), hidden_all[::1])):
        cell = get_cell_from_model(model)
        pop = PropagationOperator_LinearForm(cell, inputs, hidden)
        kop = ParameterOperator(cell, inputs, hidden)
        fop = pop @ kop

        match = re.search(r"init_([0-9.]+)", checkpoints[idx])
        g_val = float(match.group(1))

        class GetHidden(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, x):
                return self.net(x)[1]

        model_hidden = GetHidden(model)
        phi_op = pop @ kop @ pop.T()
        phi_op = HiddenNTKOperator(model_hidden, inputs, hidden)

        q = torch.randn(1, *hidden.shape)
        pf = fop.batched_call(q)
        pphi = phi_op.batched_call(q)

        sim = (pf * pphi).mean((3)) / ((pf * pf).mean((3)) * (pphi * pphi).mean((3)))**0.5

        fig.add_subplot(gs[idx, 1])
        plt.title('Random Input')
        plt.plot(q[0, :10, :, 0].T)

        fig.add_subplot(gs[idx, 2])
        plt.title(f'g = {g_val:.3f}\nOutput Under P K')
        plt.plot(pf[0, :10, :, 0].T)

        fig.add_subplot(gs[idx, 3])
        plt.title('Output Under P K P*')
        plt.plot(pphi[0, :10, :, 0].T)

        fig.add_subplot(gs[idx, 4])
        plt.title('Similarity')
#        plt.plot(sim[0, :10, :].T)
        plt.plot(sim[0, :].mean(1))
        plt.ylim(-1.1, 1.1)

        plt.tight_layout()

    plt.show()

    print('Beginning analysis with KP-Flow Operators...')
    for idx, (model, hidden) in enumerate(zip(tqdm(models[::1]), hidden_all[::1])):
        sv, sfuns = [], []
        cell = get_cell_from_model(model)

        pop = PropagationOperator_LinearForm(cell, inputs, hidden)

        class GetHidden(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, x):
                return self.net(x)[1]

        model_hidden = GetHidden(model)
        param_op = HiddenNTKOperator(model_hidden, inputs, hidden)

        functional_op = pop @ pop.T()

        functional_singular_vals = functional_op.svd(30)
        param_singular_vals = param_op.svd(30)
        
        plt.plot(functional_singular_vals)
        plt.plot(param_singular_vals)
        plt.legend(['$\\mathcal{P P^*}$, $\\mathcal{P K P^*}$'])
        plt.show()

