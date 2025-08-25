# Timing and tests for trace estimation.

# Analysis of the eigenfunction structure (modes) of operators considered.

from tqdm import tqdm
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

def parse_arguments(parser = None):
    import argparse
    parser = argparse.ArgumentParser(description='Tests and timing for trace estimation.') if parser is None else parser
    parser.add_argument('--time', action='store_true', help='Run Timing of Methods')
    parser.add_argument('--error', action='store_true', help='Run Error Measurements')
    args = parser.parse_args()
    if not (args.error or args.time):
        parser.print_help()
    return args

def rel_error(x, y):
    return np.abs(x - y) / max(np.abs(x), np.abs(y))

def time_methods(model = 'rnn'):
    from kpflow.trace_estimation import trace_hupp, trace_hupp_adj_only, trace_hupp_op, op_alignment
    from kpflow.tasks import CustomTaskWrapper
    from kpflow.architecture import Model, get_cell_from_model
    from kpflow.parameter_op import ParameterOperator, JThetaOperator
    from kpflow.propagation_op import PropagationOperator_DirectForm, PropagationOperator_LinearForm
    from kpflow.grad_op import HiddenNTKOperator

    task = CustomTaskWrapper('flip_flop', 100, use_noise = False, n_samples = 100, T = 30)
    inputs, targets = task()
    n_in, n_out = inputs.shape[-1], targets.shape[-1]
    model = Model(input_size = n_in, output_size = n_out, rnn=nn.GRU if model == 'gru' else nn.RNN, hidden_size = 256)
    model2 = Model(input_size = n_in, output_size = n_out, rnn=nn.GRU if model == 'gru' else nn.RNN, hidden_size = 256)
    out, hidden = model(inputs)
    
    class Select(nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index

        def forward(self, x):
            return x[self.index]

    model_hidden = nn.Sequential(model, Select(1)) # Select only hidden state, not output.
    model_hidden2 = nn.Sequential(model2, Select(1)) # Select only hidden state, not output.
    gop_full = HiddenNTKOperator(model_hidden, inputs, hidden, 'cpu')
    gop_full2 = HiddenNTKOperator(model_hidden2, inputs, hidden, 'cpu')

    cos_hupp = lambda A, B, nsamp: trace_hupp_op(A.T() @ B, nsamp) / (trace_hupp_op(A.T() @ A, nsamp) * trace_hupp_op(B.T() @ B, nsamp))**0.5

    nsamps = (10**np.linspace(1, 3, 40)).astype(int)
    plt.figure(figsize = (3*4, 3))
    for trace_fun in [op_alignment, cos_hupp]:
        try:
            times, evals = np.load(f'data_cross_trace_{trace_fun.__name__}.npy')
        except:
            times, evals = [], []
            for nsamp in tqdm(nsamps):
                time_0 = time.perf_counter()
                evals.append(trace_fun(gop_full, gop_full2, nsamp))
                times.append(time.perf_counter() - time_0)
            evals = np.stack(evals)
            np.save(f'data_cross_trace_{trace_fun.__name__}', (times, evals))

        plt.subplot(1,3,1)
        plt.plot(nsamps, times)
        plt.xscale('log')
        plt.xlabel('# of Trace Estimation Samples', fontsize = 14)
        plt.ylabel('Runtime', fontsize = 14)
        plt.subplot(1,3,2)
        plt.plot(nsamps, evals)
        plt.xscale('log')
        plt.ylabel('Estimated Trace', fontsize = 14)
        plt.subplot(1,3,3)
        rel_err = np.abs(evals - evals[-1]) / np.abs(evals[-1])
        plt.loglog(nsamps, rel_err)
        plt.ylabel('Relative Error vs Best Estimate', fontsize = 14)

    plt.tight_layout()
    plt.show()


def test_methods(d = 10000): 
    from kpflow.trace_estimation import trace_hupp_op
    from kpflow.op_common import MatrixWrapper
    A = np.zeros((d,d))
    nsamps = (10**np.linspace(1, 3, 40)).astype(int)

    plt.figure(figsize = (4 * 3, 3))
    for idx, k in enumerate(tqdm([3, 100, 1000])):
        U = np.random.normal(size=(d,k))
        V = np.random.normal(size=(d,k))
        A = U @ V.T
        A_op = MatrixWrapper(A)

        guesses = []
        for nsamp in nsamps:
            guesses.append(trace_hupp_op(A_op, nsamp).item())
        rel_err = np.abs(np.stack(guesses) - np.trace(A)) / np.abs(np.trace(A))

        plt.subplot(1,3,1+idx)
        plt.loglog(rel_err)
        plt.title(f'Approx Rank = {k}')
        if idx == 0:
            plt.xlabel('# of Estimator Samples', fontsize = 14)
            plt.ylabel('Relative Error', fontsize = 14)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    args = parse_arguments()
    if args.error:
        test_methods()
    if args.time:
        time_methods()
