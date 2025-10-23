from kpflow.tasks import CustomTaskWrapper
from kpflow.analysis_utils import ping_dir, load_checkpoints, import_checkpoint, torch_to_np, np_to_torch, cos_similarity
from kpflow.architecture import Model, get_cell_from_model
from kpflow.parameter_op import ParameterOperator, JThetaOperator
from kpflow.propagation_op import PropagationOperator_DirectForm, PropagationOperator_LinearForm
from kpflow.grad_op import HiddenNTKOperator
from kpflow.op_common import AveragedOperator, Operator
from kpflow.trace_estimation import op_alignment, trace_hupp_op, op_alignment_variant_1, op_alignment_variant_2

import torch, glob, time
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse, re
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA

from common import project, plot_trajectories, compute_svs, set_mpl_defaults

from tqdm import tqdm

def parse_arguments(parser = None):
    parser = argparse.ArgumentParser(description='Analyze Model on Memory Pro Task') if parser is None else parser
    parser.add_argument('--model', default='gru', type = str, help='Model to use')
    parser.add_argument('--task_str', default = 'memory_pro', type = str, help = 'Task to train on. Options: memory_pro, flip_flop, context_integration')
    parser.add_argument('--run_name', type = str)
    parser.add_argument('--recompute', action = 'store_true', help = 'Forces the code to re-compute the misdirection, which is stored and re-used on future code runs otherwise.')
    return parser.parse_args()

def plot_results(alignment_all, g_vals, losses_all):
    plt.figure(figsize = (12, 3))
    sort_inds = np.argsort(g_vals)
    g_vals = g_vals[sort_inds]
#    alignment_all = alignment_all[sort_inds]
#    g_vals = g_vals[::2]
#    alignment_all = (alignment_all[::2] + alignment_all[1::2])*.5
    losses_all = (losses_all[::2] + losses_all[1::2])*.5
    print(alignment_all.shape)

    plt.plot(g_vals, alignment_all[:,:])
    plt.show()

    n_plot =  alignment_all.shape[0] // 2
    for i in range(n_plot):
        plt.subplot(n_plot // 5, 5, i + 1)
        plt.plot(alignment_all[2*i])
        plt.plot(alignment_all[2*i+1])
        plt.ylim(-1, 1)
        plt.title(f'g = {g_vals[2*i]:.2f}')

    plt.figure()
    g0, gm = np.min(g_vals), np.max(g_vals)
    g0, gm = np.log(g0), np.log(gm)
    cmap = plt.get_cmap('gist_rainbow')
    plt.subplot(2,1,1)
    for idx in range(0, alignment_all.shape[0], 1):
        g = np.log(g_vals[idx])
        plt.plot(alignment_all[idx, :], color = cmap((g - g0) / (gm - g0)), linewidth = 0.5)
    plt.ylim(-1, 1)

    plt.subplot(2,1,2)
    for idx in range(0, alignment_all.shape[0], 1):
        g = np.log(g_vals[idx])
        plt.plot(losses_all[idx, :], color = cmap((g - g0) / (gm - g0)), linewidth = 0.5)
    plt.yscale('log')

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(alignment_all, cmap = 'hot')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(np.log(losses_all), cmap = 'hot')
    plt.colorbar()

    plt.figure()
    cmap = plt.get_cmap('cool')
    for gd_itr in [0, 50]: #[0, 10, 20, 30, 40, 50]:
        plt.plot(g_vals, alignment_all[:, gd_itr], color = cmap(gd_itr / 50), marker = 'o')
    plt.xlabel('g')
    plt.ylabel('Misdirection')

def evaluate_misdirection(args):
    task = CustomTaskWrapper(args.task_str, 50, use_noise = False, n_samples = 50, T = 90) # Uniform inputs without randomization.
    inputs, targets = task()

    n_in, n_out = inputs.shape[-1], targets.shape[-1]

    alignment_all = []
    files = []
    g_vals = []
    subdirs = glob.glob(f'{args.run_name}/*')
    for subdir in subdirs:
        file_str = f'{subdir}/repeat_0/'
        match = re.search(r"init_([0-9.]+)", file_str)
        if match is None:
            continue # Doesn's match the init_.../repeat_0/ format. 
        files.append(file_str)
        g_vals.append(float(match.group(1)))
    g_vals = np.stack(g_vals)

    sort_inds = np.argsort(g_vals)
    g_vals = g_vals[sort_inds]
    files = [files[idx] for idx in sort_inds]

#    g_vals = g_vals[[0, 10, 20, 30, 40, -8, -6, -4, -2]]
    g_vals = g_vals[::1]
    np.save(args.run_name + '/g_vals_2.npy', g_vals)

    dev = 'cpu'
    inputs, targets = inputs.to(dev), targets.to(dev)
    bad_count = 0
    g_vals = []
    losses_all = []
#    files_sub = [files[0], files[10], files[20], files[30], files[40]] + files[-8::2]
    files_sub = files[::1]
    mins = []
    for file_idx, filename in enumerate(tqdm(files_sub)):
        if filename is None:
            continue
        checkpoints, gd_itr = load_checkpoints(filename)
        if len(checkpoints) < 1:
            continue 

        print(f'Re-Evaluating {len(checkpoints)} Snapshots in {filename}...')

        try:
            g_val = float(re.search(r"init_([0-9.]+)", filename).group(1)) # Scale should be after init_ in the filename. 
        except:
            g_val = 1.

        test_losses, models, hidden_all = [], [], []
        for ch in checkpoints:
            model = Model(input_size = n_in, output_size = n_out, rnn=nn.GRU if args.model == 'gru' else nn.RNN, hidden_size = 256)
            model.load_state_dict(import_checkpoint(ch)['model'])
            model = model.to(dev)
            out, hidden = model(inputs)
            test_losses.append(nn.MSELoss()(out, targets).item())
            hidden_all.append(hidden.detach().cpu())
            models.append(model.cpu())
        hidden_all = torch.stack(hidden_all[:1])
        plt.plot(test_losses)
        test_losses = np.array(test_losses)
        mins.append(np.min(test_losses))
#        if mins[-1] > 0.002:
#            continue

        losses_all.append(test_losses[:1])
        models = models[:1]
        print(f'Hidden shape over all GD snapshots has shape {hidden_all.shape} = (GD Iter, Trial, Time, Hidden Unit)')

        alignment = []
        for idx, (model, hidden) in enumerate(zip(models, hidden_all)):
            class GetHidden(nn.Module):
                def __init__(self, net):
                    super().__init__()
                    self.net = net

                def forward(self, x):
                    return self.net(x)[1]

            if args.model == 'gru':
                torch.backends.cudnn.enabled = False      # Since GRU fused impl not implemented with vmap batching with vjp.

            model = model.to(dev)
            hidden = hidden.to(dev)
            model_hidden = GetHidden(model)
            cell = get_cell_from_model(model)
            pop = PropagationOperator_LinearForm(cell, inputs, hidden, dev = dev)
            kop = ParameterOperator(cell, inputs, hidden, dev = dev)
            gop = HiddenNTKOperator(model_hidden, inputs, hidden, dev = dev)

#            t0 = time.time()
#            tr1, tr2, tr3 = trace_hupp_op(gop.T() @ pop @ pop.T(),nsamp=30), trace_hupp_op(gop.T() @ gop,nsamp=30), trace_hupp_op(pop @ pop.T() @ pop @ pop.T(),nsamp=30)
#            t1 = time.time()
#            estimate = (tr1 / (tr2 * tr3)**0.5).item()
#            jop = gop.jtheta_op
#            jop.vectorize = True
            estimate = op_alignment(gop, pop @ pop.T(), nsamp = 30).cpu().item()
#            t2 = time.time()
#            print(t1 - t0, t2 - t1)
#            print(estimate, estimate2)

            if np.abs(estimate) > 1:
                print(f'BAD estimate count: {bad_count}')

            alignment.append(estimate)

        alignment_all.append(alignment)
        g_vals.append(g_val)
        np.save(args.run_name + '/g_vals.npy', np.array(g_vals))
        np.save(args.run_name + '/losses_all.npy', np.array(losses_all))
        np.save(args.run_name + '/alignment_all.npy', np.array(alignment_all))

    plt.figure()
    plt.plot(g_vals, mins)
    plt.figure()
    return np.array(alignment_all), np.array(g_vals), np.array(losses_all)

if __name__ == '__main__':
    PALLETTE = ['#f86969ff', '#7e69f8ff', '#f8c969ff', '#69f87cff', '#e569f8ff']
    set_mpl_defaults(15)
    args = parse_arguments()
    print(f'Task string {args.task_str}')
    try:
        assert(not args.recompute) # Force re-run.
        alignment_all = np.load(args.run_name + '/alignment_all.npy')
        g_vals = np.load(args.run_name + '/g_vals.npy')
        losses_all = np.load(args.run_name + '/losses_all.npy')
    except:
        print("Re-evaluating misdirection. May take a while...")
        alignment_all, g_vals, losses_all = evaluate_misdirection(args)
        alignment_all = np.load(args.run_name + '/alignment_all.npy')
        g_vals = np.load(args.run_name + '/g_vals.npy')
        losses_all = np.load(args.run_name + '/losses_all.npy')

    plot_results(alignment_all, g_vals, losses_all)
    plt.show()


# EVALUATION OF METHODS (TODO MOVE):
#            import time
#            baseline = trace_hupp_op(gop.T() @ pop @ pop.T(),nsamp=30) / (trace_hupp_op(gop.T() @ gop,nsamp=30) * trace_hupp_op(pop @ pop.T() @ pop @ pop.T(),nsamp=30))**0.5
#            t0 = time.time()
#            estimates = op_alignment(gop, pop @ pop.T(), nsamp = 30, full_output = True)
#            t1 = time.time()
#
#            inputs = inputs.cuda()
#            hidden = torch.from_numpy(hidden).cuda()
#            cell = cell.cuda()
#            model_hidden.net = model_hidden.net.cuda()
#            pop = PropagationOperator_LinearForm(cell, inputs, hidden, dev = 'cuda')
#            gop = HiddenNTKOperator(model_hidden, inputs, hidden, dev = 'cuda')
#
#            t2 = time.time()
#            jop = gop.jtheta_op
#            jop.vectorize = True
#            estimates2 = op_alignment_variant_1(jop, jop, pop @ pop.T(), nsamp = 100, full_output = True).cpu()
#            t3 = time.time()
#            estimates3 = op_alignment_variant_2(jop, jop, pop @ pop.T(), nsamp = 100, full_output = True).cpu()
#            t4 = time.time()
#            print(t1 - t0, t3 - t2, t4 - t3)
#
#            plt.plot(np.abs(estimates - baseline) / np.abs(estimates[-1]))
#            plt.plot(np.abs(estimates2 - baseline) / np.abs(estimates2[-1]))
#            plt.plot(np.abs(estimates3 - baseline) / np.abs(estimates2[-1]))
#            plt.ylabel('Relative Error')
#            plt.yscale('log')
#
#            plt.figure()
#            plt.axhline(baseline, c = 'black', linestyle = 'dashed')
#            plt.plot(estimates)
#            plt.plot(estimates2)
#            plt.plot(estimates3)
#            plt.show()
