# Perform timing on the trace estimation methods described in the main paper.
import numpy as np
import torch, torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from kpflow.op_common import Operator
from kpflow.grad_op import FullJThetaOperator as JVP, HiddenNTKOperator as NTK
import argparse
from time import perf_counter as pf
from torch.nn.utils import parameters_to_vector as p2v
import matplotlib.ticker as mticker

from train_mlps import train_once, get_loaders, MLP as MLP2
from trace_methods import *

import sys, os
sys.path.append('../')
from common import set_mpl_defaults, ping_dir, annotate_subplots

def parse_args():
    parser = argparse.ArgumentParser(description='Perform timing on trace estimation methods described in main paper.')
    parser.add_argument('--rerun', action = 'store_true', help = 'Whether to rerun experiments measuring timing.')
    parser.add_argument('--dev', default = 'cpu', type = str, help = 'cuda or cpu')
    return parser.parse_args()

def get_operators_gru(B, T, H, seed = 0, dev = 'cpu'): # Batch count, GRU timesteps, hidden count
    class GRU(nn.Module):
        def __init__(self, n_in, H):
            super().__init__()
            self.rnn = nn.GRU(n_in, H)

        def forward(self, x):
            return self.rnn(x)[0]

    n_in = 10
    torch.manual_seed(seed)
    x = torch.randn((B, T, n_in)).to(dev) # Random input.
    model = GRU(n_in, H).to(dev)
    hidden = model(x)
    total_params = sum(p.numel() for p in model.parameters())
    print("# params = ", total_params)

    jvp = JVP(model, x, hidden, dev = dev, params_to_vec = True)
    vjp = jvp.T
    ntk = jvp @ vjp
    return vjp, jvp, ntk

def get_operators_mlp(B, T, H, seed = 0, dev = 'cpu'): # Batch count, MLP layers, hidden count
    class MLP(nn.Module):
        def __init__(self, n_in, H):
            super().__init__()
            self.layers = nn.Sequential(nn.Linear(n_in, H), 
                *[nn.Sequential(nn.ReLU(), nn.Linear(H, H)) for _ in range(T-1)],
                nn.ReLU()
            )

        def forward(self, x):
            return self.layers(x)

    n_in = 100 
    torch.manual_seed(seed)
    x = torch.randn((B, n_in)).to(dev) # Random input.
    model = MLP(n_in, H).to(dev)
    out = model(x)
    total_params = sum(p.numel() for p in model.parameters())
    print("# params = ", total_params)

    jvp = JVP(model, x, out, dev = dev, params_to_vec = True)
    vjp = jvp.T
    ntk = jvp @ vjp
    return vjp, jvp, ntk

def get_operators_mlp_trained(state_dict, dev = 'cpu'):
    model = MLP2(hidden=256)
    model.load_state_dict(state_dict)
    model = model.to(dev)
    for x, _ in get_loaders()[1]:
        break 

    out = model(x)

    jvp = JVP(model, x, out, dev = dev, params_to_vec = True)
    vjp = jvp.T
    ntk = jvp @ vjp
    return vjp, jvp, ntk

def timing(baseline, methods, reruns, nsamps):
    # baseline is a function giving the true value of the quantity we want to estimate.
    # methods should be a dict with entries 'name': fn, where fn takes in nsamp and produces a result.
    ret = {}

    ret_shape = (reruns, nsamps.shape[0])
    ret['estimates'] = {name: np.zeros(ret_shape) for name in methods.keys()} 
    ret['times'] = {name: np.zeros(ret_shape) for name in methods.keys()} 
    ret['nsamps'] = nsamps

    time0 = pf()
    ret['baseline'] = baseline()
    ret['time_baseline'] = pf() - time0
    print(ret['time_baseline'])

    print(f"Performing timing with {reruns} Re-runs and nsamp in range [{np.min(nsamps)}, {np.max(nsamps)}]")
    for i in tqdm(range(reruns)):
        for k, nsamp in enumerate(nsamps):
            for name, fn in methods.items():
                time0 = pf()
                ret['estimates'][name][i,k] = fn(nsamp)
                ret['times'][name][i,k] = pf() - time0
                print(nsamp, (ret['times'][name][i,k] / ret['time_baseline']) * 100)

    return ret

def time_to_err(time, err, true, eps):
    try:
        i2 = int(np.argwhere(err <= eps)[0])
    except IndexError:
        return np.nan

    i1 = i2 - 1 

    t1 = 0. if i2 == 0 else time[i1]
    t2 = time[i2]
    err1 = np.abs(true) if i2 == 0 else err[i1]
    err2 = err[i2]

    t = ((eps - err1) / (err2 - err1)) * (t2 - t1) + t1 # Linearly interpolate t1 and t2.
    return t

def get_data(fname, ref = None):
    nsamps = np.load(f'data/{fname}/nsamps.npy', allow_pickle = True)
    times = np.load(f'data/{fname}/times.npy', allow_pickle = True).item()
    vals = np.load(f'data/{fname}/vals.npy', allow_pickle = True).item()
    if ref is not None:
        vals['Direct']  = 0.* vals['Direct'] + ref
    ref = vals['Direct'][0, -1]
    errs = {name: np.abs(val - vals['Direct']) / np.abs(val) for name, val in vals.items()}
    return nsamps, times, vals, ref, errs

def plot_results(model):
    set_mpl_defaults(15) # Pretty matplotlib plots.
    colors = [None, *plt.rcParams['axes.prop_cycle'].by_key()['color']]

    title = f'Estimating Trace for {model.upper()}\n'

    nsamps, times, vals, ref, errs = get_data(f'{model}_big_n')

    fig1 = plt.figure(figsize = (4*3*1.2, 3*1*1.2))
    plt.subplot(1, 3, 1)
    for name, color in zip(times.keys(), colors):
        if name == 'Direct':
            continue
        v = vals[name]
        median = np.median(v, axis=0)               
        q25 = np.percentile(v, 25, axis=0)            
        q75 = np.percentile(v, 75, axis=0)
        med_time = np.median(times[name], axis=0)
#        plt.plot(med_time, median, label = name, color = color)
        plt.plot(nsamps, median, label = name, color = color)
#        plt.fill_between(med_time, q25, q75, facecolor = color, label = '_nolabel_', alpha = .5)
        plt.fill_between(nsamps, q25, q75, facecolor = color, label = '_nolabel_', alpha = .5)

    plt.axhline(ref, color = 'black', zorder = -1, alpha = .5, linestyle = 'dashed', label = 'True')
    plt.title('Trace Estimate')#, $t$')
    plt.xlabel('# of Samples, $m$')
    plt.xscale('log')
    plt.grid(True, alpha = 0.25)
#    plt.yscale('log')
#    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{v:.0f}"))
#    plt.gca().yaxis.set_minor_formatter(mticker.FuncFormatter(lambda v, pos: f"{v:.0f}"))
    plt.legend(frameon = False, ncol = 2)

    plt.subplot(1, 3, 2)
    for name, color in zip(times.keys(), colors):
        # times[name] is shape (d1, d2) where d1 is count of Hs, d2 is count of nsamps.
        err = errs[name]
        median = np.median(err, axis=0)               
        q25 = np.percentile(err, 25, axis=0)            
        q75 = np.percentile(err, 75, axis=0)
        times_mean = times[name].mean(0)

        if name != 'Direct':
            plt.plot(times_mean, median, label = name, color = color)
            plt.fill_between(times_mean, q25, q75, facecolor = color, label = '_nolabel_', alpha = .5)

    plt.ylabel('Relative Error')#, $\\frac{|\\text{tr}(\\text{NTK}) - t|}{|t|}$')
    plt.xlabel('Runtime (s)')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, alpha = 0.25)
    plt.title(title)

    # Zoom in small nsamp range. Still use old reference value.
    nsamps, times, vals, _, errs = get_data(f'{model}_small_n', ref = ref)

    plt.subplot(1, 3, 3)
    for name, color in zip(times.keys(), colors):
        # times[name] is shape (d1, d2) where d1 is count of Hs, d2 is count of nsamps.
        err = errs[name]
        median = np.median(err, axis=0)               
        q25 = np.percentile(err, 25, axis=0)            
        q75 = np.percentile(err, 75, axis=0)
        times_mean = times[name].mean(0)

        if name != 'Direct':
            plt.plot(times_mean, median, label = name, color = color)
            plt.fill_between(times_mean, q25, q75, facecolor = color, label = '_nolabel_', alpha = .5)

    plt.ylabel('Relative Error')#, $\\frac{|\\text{tr}(\\text{NTK}) - t|}{|t|}$')
    plt.xlabel('Runtime (s)')
    plt.yscale('log')
    plt.grid(True, alpha = 0.25)
#    plt.xscale('log')

    annotate_subplots()
    plt.tight_layout()

def plot_err_bar(xdata, data, percentile_range, color, label = '', alpha = .5):
    # Assume samples are in in axis 0.
    median = np.nanmedian(data, axis=0)               
    qmin = np.percentile(data, 50 - percentile_range, axis=0)            
    qmax = np.percentile(data, 50 + percentile_range, axis=0)

    plt.plot(xdata, median, label = label, color = color)
    plt.fill_between(xdata, qmin, qmax, facecolor = color, label = '_nolabel_', alpha = .5)

def plot_speedup():
    set_mpl_defaults(15) # Pretty matplotlib plots.
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.figure(figsize = (5*2*1.2, 3*1*1.2))
    for idx, model in enumerate(['mlp', 'gru']):
        plt.subplot(1,2,idx+1)
        nsamps, times, vals, ref, errs = get_data(f'{model}_big_n')

        sub_names = list(times.keys())
        sub_names.remove('Direct')
        tols = 10**np.linspace(-2, -5, 20)
        speedups = []
        print("Baseline time", times['Direct'].mean())

        for name in sub_names:
            speedups.append([])
            for time_run, err_run in zip(times[name], errs[name]):
                time_to_tol = np.stack([time_to_err(time_run, err_run, ref, eps) for eps in tols])
                speedups[-1].append(times['Direct'].mean() / time_to_tol)

        speedups_all = np.array(speedups) # (3, reruns, len(tols))
        speedups = np.nanmax(speedups_all, 0) # Pick one that did best for each rerun
        print(np.round(np.mean(np.nanargmax(speedups_all, 0),0)).astype(int))

        acc = 100 * (1 - tols)
        plot_err_bar(tols, speedups, 25, colors[5 + 4 * idx], '')

        plt.xscale('log')
        plt.yscale('log')

        plt.gca().invert_xaxis()
        plt.gca().xaxis.set_minor_locator(plt.FixedLocator([]))
        plt.gca().set_xticks([1e-2, 1e-3, 1e-5]) 
        plt.gca().set_xticklabels(['99%', '99.9%', '99.999%'])
        if idx == 0:
            plt.ylabel('Relative Speedup')
        plt.xlabel('Accuracy')

        my_n = '3.2k' if idx == 0 else '48k'
        my_p = '64.7k' if idx == 0 else '14.5k'
        plt.title(f'Example {1+idx}\n{model.upper()} Model (n={my_n}, n$_\\theta$={my_p})')

    plt.suptitle('Estimating $\\text{tr}(\\text{NTK})$: Speedup v. Accuracy', y=0.92)  # move it closer; default is ~0.98
    annotate_subplots()
    plt.tight_layout()

def plot_extras():
    set_mpl_defaults(15) # Pretty matplotlib plots.
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    data_runs = {
        'Norm': np.load('data/special_terms/ret_norm.npy', allow_pickle = True).item(),
        'Alignment': np.load('data/special_terms/ret_cos.npy', allow_pickle = True).item(),
        'Effective Rank': np.load('data/special_terms/ret_effrank.npy', allow_pickle = True).item()
    }
    plt.figure(figsize = (4*3*1.2, 3*1*1.2))
    for idx, (name, data_run) in enumerate(data_runs.items()):
        ref = data_run['baseline']
        print(data_run['nsamps'])
        times, vals = data_run['times']['Hutch++'], data_run['estimates']['Hutch++'] # Each (reruns, #nsamps)
        times, vals = times[:, :-3], vals[:, :-3]
        med_time = np.median(times, 0)
        baseline_time = data_run['time_baseline']
        print(baseline_time)
        errs = np.abs(vals - ref.item()) / np.abs(vals)

        plt.subplot(1,3,1+idx)
        time_percent = (med_time / data_run['time_baseline'])
        plot_err_bar(time_percent, errs, 25, colors[0], label = 'Hutch++' )
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{100*val:.1f}%'))
        if idx == 0:
            plt.legend(frameon = False)

        plt.xlabel('Relative Runtime (%)')
        plt.ylabel('Relative Error')

        plt.title(name)
        plt.yscale('log')

#        tols = 10**np.linspace(-1, -4, 20)
#        speedups = []
#        for time_run, err_run in zip(times, errs):
#            time_to_tol = np.stack([time_to_err(time_run, err_run, ref, eps) for eps in tols])
#            speedups.append(baseline_time / time_to_tol)
#
#        speedups = np.array(speedups) # (reruns, len(tols))
#
#        acc = 100 * (1 - tols)
#        plot_err_bar(tols, speedups, 25, colors[0], '')
#
#        plt.xscale('log')
#        plt.yscale('log')
#
#        plt.gca().invert_xaxis()
#        plt.gca().xaxis.set_minor_locator(plt.FixedLocator([]))
#        plt.gca().set_xticks([1e-1, 1e-2, 1e-4]) 
#        plt.gca().set_xticklabels(['90%', '99%', '99.99%'])
#        if idx == 0:
#            plt.ylabel('Relative Speedup')
#        plt.xlabel('Accuracy')

#        my_n = '3.2k' if idx == 0 else '48k'
#        plt.title(f'{model.upper()} Model (n={my_n})')

    annotate_subplots()
    plt.suptitle('Estimating Related Terms (n=5.12k)')
    plt.tight_layout()


def cos_trace_def(ntk0, ntk1, trace_fn):
    return trace_fn(ntk0 @ ntk1.T) / (trace_fn(ntk0.gram()) * trace_fn(ntk1.gram()))**.5

def cos_hupp_op(ntk0, ntk1, nsamp):
    return cos_trace_def(ntk0, ntk1, lambda x: trace_hupp_op(x, nsamp=nsamp))

def cos_direct(ntk0, ntk1, nsamp):
    return cos_trace_def(ntk0, ntk1, trace_direct)

def timing_old(args, vjp, jvp, ntk, reruns, nsamps, dev, compute_direct = False):
    ret_shape = (reruns, nsamps.shape[0])
    vals = {name: np.zeros(ret_shape) for name in ['Direct', 'Hutch++', 'RHutch', 'FHutch']} 
    times = {name: np.zeros(ret_shape) for name in ['Direct', 'Hutch++', 'RHutch', 'FHutch']} 

    time0 = pf()
    if compute_direct:
        vals['Direct'] += trace_direct(ntk)
    time1 = pf()
    times['Direct'] += time1 - time0
    print(time1 - time0)

    for i in tqdm(range(reruns)):
        for k, nsamp in enumerate(nsamps):
            # First compute the true trace by n matvecs. This will take a while...
            time0 = pf()
            vals['Hutch++'][i,k] = trace_hupp_op(ntk, nsamp = nsamp)
            time1 = pf()
            vals['RHutch'][i,k] = trace_one_sided(vjp, nsamp = nsamp)
            time2 = pf()
            vals['FHutch'][i,k] = trace_one_sided(jvp, nsamp = nsamp)
            time3 = pf()
            times['Hutch++'][i,k] = time1 - time0
            times['RHutch'][i,k] = time2 - time1
            times['FHutch'][i,k] = time3 - time2

    return nsamps, times, vals

def extra_term_experiments(args, reruns, nsamps):
    sd0, sdf = torch.load('data/mlps_mnist/model_init_0.pt', weights_only=True), torch.load('data/mlps_mnist/model_final_0.pt', weights_only=True)

    vjp0, jvp0, ntk0 = get_operators_mlp_trained(sd0, args.dev)
    vjpf, jvpf, ntkf = get_operators_mlp_trained(sdf, args.dev)

    est_direct = lambda A, B : trace_direct(A @ B.T)
    est_hupp = lambda A, B, nsamp : trace_hupp_op(A @ B.T, nsamp = nsamp)

    # Norm.
    ret_norm = timing(
        lambda : est_direct(ntkf, ntkf),
        {
            'Hutch++': lambda nsamp : est_hupp(ntkf, ntkf, nsamp),
        }, 
        reruns, nsamps
    )

    # Cosine similarity.
    ret_cos = timing(
        lambda : est_direct(ntkf, ntk0) / (est_direct(ntkf, ntkf) * est_direct(ntk0, ntk0))**.5,
        {
            'Hutch++': lambda nsamp : est_hupp(ntkf, ntk0, nsamp) / (est_hupp(ntkf, ntkf, nsamp) * est_hupp(ntk0, ntk0, nsamp))**.5
        }, 
        reruns, nsamps
    )

    # Effective rank.
    ret_effrank = timing(
        lambda : trace_direct(ntkf)**2 / trace_direct(ntkf @ ntkf),
        {
            'Hutch++': lambda nsamp : trace_hupp_op(ntkf, nsamp=nsamp)**2 / trace_hupp_op(ntkf @ ntkf, nsamp=nsamp)
        }, 
        reruns, nsamps
    )
    return ret_norm, ret_cos, ret_effrank

def run_experiments(args):
    # Store results here:
    ping_dir('data/')
    ping_dir('data/gru_small_n/')
    ping_dir('data/gru_big_n/')
    ping_dir('data/mlp_small_n/')
    ping_dir('data/mlp_big_n/')
    ping_dir('data/special_terms/')

    # mlp big nsamp
    print("Evaluating MLP Trace Estimates for Big m")
    reruns = 50
    nsamps = (10**np.linspace(1, np.log10(3200/6), 10) * 6).astype(int) # nsamp should be a multiple of 6
    vjp, jvp, ntk = get_operators_mlp(B=50, T=15, H=64) # Get all the operators involved :)
    nsamps, times, vals = timing_old(args, vjp, jvp, ntk, reruns, nsamps, dev=args.dev, compute_direct = True)

    np.save('data/mlp_big_n/nsamps.npy', nsamps)
    np.save('data/mlp_big_n/times.npy', times)
    np.save('data/mlp_big_n/vals.npy', vals)
    print("Saved experimental data for MLP big nsamp")

    # mlp small nsamp
    print("Evaluating MLP Trace Estimates for Small m")
    reruns = 50 
    nsamps = 6 * np.linspace(1, 50, 20).astype(int)
    vjp, jvp, ntk = get_operators_mlp(B=50, T=15, H=64) # Get all the operators involved :)
    nsamps, times, vals = timing_old(args, vjp, jvp, ntk, reruns, nsamps, dev=args.dev)

    np.save('data/mlp_small_n/nsamps.npy', nsamps)
    np.save('data/mlp_small_n/times.npy', times)
    np.save('data/mlp_small_n/vals.npy', vals)
    print("Saved experimental data for MLP small nsamp")

    # gru big nsamp
    print("Evaluating GRU Trace Estimates for Big m")
    reruns = 50
    nsamps = (10**np.linspace(1, np.log10(2000), 15) * 6).astype(int) # nsamp should be a multiple of 6
    vjp, jvp, ntk = get_operators_gru(B=50, T=15, H=64) # Get all the operators involved :)
    nsamps, times, vals = timing_old(args, vjp, jvp, ntk, reruns, nsamps, dev=args.dev, compute_direct = True)

    np.save('data/gru_big_n/nsamps.npy', nsamps)
    np.save('data/gru_big_n/times.npy', times)
    np.save('data/gru_big_n/vals.npy', vals)
    print("Saved experimental data for GRU big nsamp")

    # gru small nsamp
    print("Evaluating GRU Trace Estimates for Small m")
    reruns = 50 
    nsamps = 6 * np.linspace(1, 50, 20).astype(int)
    vjp, jvp, ntk = get_operators_gru(B=50, T=15, H=64) # Get all the operators involved :)
    nsamps, times, vals = timing_old(args, vjp, jvp, ntk, reruns, nsamps, dev=args.dev)

    np.save('data/gru_small_n/nsamps.npy', nsamps)
    np.save('data/gru_small_n/times.npy', times)
    np.save('data/gru_small_n/vals.npy', vals)
    print("Saved experimental data for GRU small nsamp")

    # mlp for other metrics.
    print("Training an MLP on MNIST")
    acc = train_once(idx=0, epochs=5, lr=1e-3, batch_size=128, hidden=256)
    print('MNIST Accuracy {acc}')

    print("Evaluating MNIST MLP: Norm, Cos, Effrank Estimates")
    reruns = 50
    nsamps = (10**np.linspace(1, np.log10(5120/6), 10) * 6).astype(int) # nsamp should be a multiple of 6
    ret_norm, ret_cos, ret_effrank = extra_term_experiments(args, reruns, nsamps)
    np.save('data/special_terms/ret_norm.npy', ret_norm)
    np.save('data/special_terms/ret_cos.npy', ret_cos)
    np.save('data/special_terms/ret_effrank.npy', ret_effrank)

def main(args):
    if args.rerun or not os.path.exists('data/'): # If not data folder, make it and run experiments from scratch.
        run_experiments(args)

    # Make plots.
    plot_speedup() # Figure 1
    plt.savefig('figures/speedup.pdf')
    plot_results('mlp') # Figure 2
    plt.savefig('figures/trace_estimate_mlp.pdf')
    plot_results('gru') # Figure 3
    plt.savefig('figures/trace_estimate_gru.pdf')
    plot_extras() # Figure 4 
    plt.savefig('figures/extras.pdf')
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    main(args)
