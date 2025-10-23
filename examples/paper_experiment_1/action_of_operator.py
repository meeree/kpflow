# Action of operator Phi on a random input, for illustration.
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
    parser = argparse.ArgumentParser(description='Talk Example') if parser is None else parser
    parser.add_argument('--model', default='gru', type = str, help='Model to use')
    parser.add_argument('--task_str', default = 'memory_pro', type = str, help = 'Task to train on. Options: memory_pro, flip_flop, context_integration')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    set_mpl_defaults(14)
    PALLETTE = ['#f86969ff', '#7e69f8ff', '#f8c969ff', '#69f87cff', '#e569f8ff']

    task = CustomTaskWrapper(args.task_str, 50, use_noise = False, n_samples = 50, T = 90) # Uniform inputs without randomization.
    inputs, targets = task()
    n_in, n_out = inputs.shape[-1], targets.shape[-1]

    model = Model(input_size = n_in, output_size = n_out, rnn=nn.GRU if args.model == 'gru' else nn.RNN, hidden_size = 256)
    out, hidden = model(inputs)

    cell = get_cell_from_model(model)
    pop = PropagationOperator_LinearForm(cell, inputs, hidden)
    kop = ParameterOperator(cell, inputs, hidden)
    fop = pop @ kop

    class GetHidden(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x)[1]

    model_hidden = GetHidden(model)
    phi_op = pop @ kop @ pop.T()
    #phi_op = HiddenNTKOperator(model_hidden, inputs, hidden)

    svals, svecs = phi_op.svd(10, compute_vecs = True)
    print(svecs.shape)

    q = torch.randn(1, *hidden.shape)
    pf = fop.batched_call(q)
    pphi = phi_op.batched_call(q)

    sim = (pf * pphi).mean((3)) / ((pf * pf).mean((3)) * (pphi * pphi).mean((3)))**0.5

    fig = plt.figure()
    plt.subplot(1, 4, 1)
    plt.title('Random Input')
    plt.plot(q[0, :10, :, 0].T)

    plt.subplot(1, 4, 2)
    plt.title('Output Under P K')
    plt.plot(pf[0, :10, :, 0].T)

    plt.subplot(1, 4, 3)
    plt.title('Output Under P K P*')
    plt.plot(pphi[0, :10, :, 0].T)

    plt.subplot(1, 4, 4)
    plt.title('Similarity')
#        plt.plot(sim[0, :10, :].T)
    plt.plot(sim[0, :].mean(1))
    plt.ylim(-1.1, 1.1)

    plt.tight_layout()

    plt.show()
