# Timing and tests for trace estimation.

# Analysis of the eigenfunction structure (modes) of operators considered.

from tqdm import tqdm
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

def measure_times(*fns):
    times = {}
    for fn in fns:
        time_0 = time.perf_counter()
        fn()
        times[fn.__name__] = time.perf_counter() - time_0
    return times

def time_methods(model = 'rnn'):
    from kpflow.trace_estimation import trace_hupp, trace_hupp_adj_only, trace_hupp_op
    from kpflow.tasks import CustomTaskWrapper
    from kpflow.architecture import Model, get_cell_from_model
    from kpflow.parameter_op import ParameterOperator, JThetaOperator
    from kpflow.propagation_op import PropagationOperator_DirectForm, PropagationOperator_LinearForm
    from kpflow.grad_op import HiddenNTKOperator

    task = CustomTaskWrapper('flip_flop', 100, use_noise = False, n_samples = 100, T = 30)
    inputs, targets = task()
    n_in, n_out = inputs.shape[-1], targets.shape[-1]
    model = Model(input_size = n_in, output_size = n_out, rnn=nn.GRU if model == 'gru' else nn.RNN, hidden_size = 256)
    out, hidden = model(inputs)
    
    class Select(nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index

        def forward(self, x):
            return x[self.index]

    model.flatten_parameters = lambda: None
    model_hidden = nn.Sequential(model, Select(1)) # Select only hidden state, not output.
    gop_full = HiddenNTKOperator(model_hidden, inputs, hidden, 'cpu')
    time_0 = time.perf_counter()
    trace_hupp_op(gop_full, 50)
    print(time.perf_counter() - time_0)

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
