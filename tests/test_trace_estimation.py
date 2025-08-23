# Timing and tests for trace estimation.

# Analysis of the eigenfunction structure (modes) of operators considered.

from tqdm import tqdm
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

def rel_error(x, y):
    return np.abs(x - y) / max(np.abs(x), np.abs(y))

def time_methods(model = 'rnn'):
    from kpflow.trace_estimation import trace_hupp, trace_hupp_adj_only, trace_hupp_op, cross_trace_op
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
    plt.figure(figsize = (4, 3))
    for trace_fun in [cross_trace_op, cos_hupp]:
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

        rel_err = np.abs(evals - evals[-1]) / np.abs(evals[-1])
        plt.plot(times[1:], rel_err[1:])
#        plt.xscale('log')
        plt.yscale('log')
#
#        plt.subplot(1,3,1)
#        plt.plot(nsamps, times)
#        plt.xscale('log')
#        plt.xlabel('# of Trace Estimation Samples', fontsize = 14)
#        plt.ylabel('Runtime', fontsize = 14)
#        plt.subplot(1,3,2)
#        plt.plot(nsamps, evals)
#        plt.xscale('log')
#        plt.ylabel('Estimated Trace', fontsize = 14)
#        plt.subplot(1,3,3)
#        rel_err = np.abs(evals - evals[-1]) / np.abs(evals[-1])
#        plt.loglog(nsamps, rel_err)
#        plt.ylabel('Relative Error vs Best Estimate', fontsize = 14)

    plt.tight_layout()
    plt.show()

    # NTK alignment.
    cell = get_cell_from_model(model)
    pop = PropagationOperator_LinearForm(cell, inputs, hidden)
    kop = ParameterOperator(cell, inputs, hidden)

    jop = JThetaOperator(cell, inputs, hidden)
    jop.vectorize = True

    g = (pop @ kop @ pop.T())

    # Test out vmap versus sequential eval.
    inp = 1e-3 * torch.ones((100, *hidden.shape))
    def vmap_approach():
        return torch.vmap(gop_full)(inp)

    def sequential_approach():
        v2 = torch.zeros_like(inp)
        for i in range(v2.shape[0]):
            v2[i] = gop_full(inp[i])
        return v2
#
#    print(measure_times(vmap_approach, sequential_approach))
#
#    asjdoisadjois

#    inp = 1e-3 * np.ones(hidden.shape)
#    v1, v2 = gop_full(inp).detach().numpy(), (pop @ kop @ pop.T())(inp).detach().numpy()
#    print(np.max(np.abs(v1 - v2)) / max(np.max(np.abs(v1)), np.max(np.abs(v2))))

#    time_0 = time.perf_counter()
#    trace_hupp_adj_only((pop @ kop @ pop.T()).to_scipy(), (pop @ jop).to_scipy(), 50)
#    print(time.perf_counter() - time_0)

    time_0 = time.perf_counter()
    trace_hupp(gop_full.to_scipy(), 50)
    print(time.perf_counter() - time_0)

    time_0 = time.perf_counter()
    trace_hupp_op(g, 50)
    print(time.perf_counter() - time_0)

time_methods()


#nsamps = (10**np.linspace(1, 3, 10)).astype(int)
#
#d = 10000
#A = np.zeros((d,d))
#
#plt.figure(figsize = (4 * 3, 3))
#for idx, k in enumerate([3, 100, 1000]):
#    U = np.random.normal(size=(d,k))
#    V = np.random.normal(size=(d,k))
#    A = U @ V.T
#
#    guesses = []
#    for nsamp in nsamps:
#        guesses.append(trace_hpp(A, nsamp))
#
#    plt.subplot(1,3,1+idx)
#    plt.plot(nsamps, guesses, color = 'blue')
#    plt.axhline(np.trace(A), color = 'red', linestyle = 'dashed')
#
#plt.show()
