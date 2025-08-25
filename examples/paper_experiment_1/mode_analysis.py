# Analysis of the eigenfunction structure (modes) of operators considered.

from kpflow.tasks import CustomTaskWrapper
from kpflow.analysis_utils import ping_dir, load_checkpoints, import_checkpoint, torch_to_np, np_to_torch, cos_similarity
from kpflow.architecture import Model, get_cell_from_model
from kpflow.parameter_op import ParameterOperator, JThetaOperator
from kpflow.propagation_op import PropagationOperator_DirectForm, PropagationOperator_LinearForm
from kpflow.op_common import AveragedOperator, Operator
from kpflow.trace_estimation import trace_hpp

from common import project, plot_trajectories, compute_svs, set_mpl_defaults

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

    print('Beginning analysis with KP-Flow Operators...')
    colors = (np.stack([ang, ang * 0., ang * 0.], -1) + np.pi) / (2 * np.pi)
    colors = colors[[0, 50, 100, 150]]
    colors = ['red', 'green', 'blue', 'purple']
    inputs, hidden_all = inputs[[0, 50, 100, 150]], hidden_all[:, [0, 50, 100, 150]]
    ping_dir(f'anim_frames_scales_{args.task_str}/')
    for idx, (model, hidden) in enumerate(zip(tqdm(models[::1]), hidden_all[::1])):
        sv, sfuns = [], []
        cell = get_cell_from_model(model)
        for i in range(hidden.shape[0]):
            jop = JThetaOperator(cell, inputs[i:i+1], hidden[i:i+1]) 
            pop = PropagationOperator_LinearForm(cell, inputs[i:i+1], hidden[i:i+1])
            kop = ParameterOperator(cell, inputs, hidden)
            gram_c = pop @ pop.T()   # Controllability Grammian

            avg_shape = (1, hidden.shape[1], 1)
            avg_gram_c = AveragedOperator(gram_c, hidden.shape)

            avg_pop = AveragedOperator(pop, hidden.shape)

            sv_i, sfuns_i = compute_svs(avg_gram_c, avg_shape, 42, True)

            plt.figure()
            for mean in np.linspace(-1., 1., 10):
                signal = np.random.normal(size = sfuns_i[0].shape, scale = 0.1) + mean
                output = avg_pop(signal)

                plt.subplot(1,3,1)
                plt.plot(signal[0, :, 0])
                plt.subplot(1,3,2)
                plt.plot(output[0, :, 0])
                plt.subplot(1,3,3)
                plt.plot(np.cumsum(signal[0, :, 0]))

            # Re-orient sfuns
            sfuns_i *= np.sign(sfuns_i[:, :, 0:1, :]) 
            sv.append(sv_i); sfuns.append(sfuns_i[:, 0])

        sv = np.stack(sv, 1)
        sfuns = np.stack(sfuns, 1)

#        plt.figure()
#        plt.plot(sv)
#        plt.xlabel('Index, $k$')
#        plt.ylabel('$\\sigma_k$, singular value')

        kop = ParameterOperator(cell, inputs, hidden)
        dim, varrat = gram_c.effrank(sv, .95) 

#        plt.figure()
#        plt.plot(sv)
#
#        norm_mean = np.mean(np.linalg.norm(sfuns, axis=-1), -1) # Norm over hidden, mean over time. 
#        plt.figure()
#        plt.plot(norm_mean[0])

        wsfuns = sv[:, :, None, None] * sfuns
        ymin, ymax = wsfuns.min(), wsfuns.max()

#        def fit_with_sine(mode, p0 = None):
#            # 1) quick FFT to estimate the main frequency
#            times = np.arange(mode.shape[0])
#            y = mode - mode.mean()                     # remove any DC component
#            dt = times[1] - times[0]
#            freqs = np.fft.rfftfreq(len(times), dt)
#            fft_mag = np.abs(np.fft.rfft(y))
#            f0 = freqs[np.argmax(fft_mag)]             # dominant frequency (Hz)
#
#            # amplitude guess = RMS * √2, phase guess = 0
#            p0 = (np.sqrt(2)*y.std(), f0, 0.0) if p0 is None else p0
#
#            # 2) non-linear least squares
#            popt, _ = curve_fit(sine, times, y, p0=p0, maxfev=50000)
#            return sine(times, *popt) + mode.mean(), popt
#
#        wsfun_mean = wsfuns.mean(1)
#        plt.figure(figsize = (12, 8))
#        popt = None
#        popts = []
#        for i in range(sv.shape[0]):
#            vals = wsfun_mean[i, :, 0]
##            popt, _ = curve_fit(sine, np.arange(vals.shape[0]), vals, p0=(1.,1.,0.))
#
#            plt.subplot(6, 7, i+1)
#            approx, popt = fit_with_sine(vals, p0 = popt)
#            popts.append(np.copy(popt))
#
#            plt.plot(approx)
#            plt.title(f'{popt[0]:.3f}, {popt[1]:.3f}, {popt[2]:.3f}')
##            plt.plot(sine(np.arange(vals.shape[0]), *popt))
#            plt.plot(vals)
##            plt.title(popt[1])
#        plt.tight_layout()
#
#        plt.figure(figsize = (12, 3))
#        popts = np.stack(popts)[1:] # Remove first mode.
#        for i in range(3):
#            plt.subplot(1,3,1+i)
#            plt.plot(popts[:, i])
#            if i == 1:
#                m, b = np.polyfit(np.arange(popts.shape[0]), popts[:, i], 1)
#                plt.plot(m * np.arange(popts.shape[0]) + b)
#                plt.title(f'{m : .3f} * x + {b : .3f}')
#            elif i == 2:
#                m, b = np.polyfit(np.arange(popts.shape[0]), 1. / popts[:, i], 1)
#                plt.plot(1. / (m * np.arange(popts.shape[0]) + b))
#        plt.tight_layout()
#        plt.show()

        plt.figure(figsize = (12, 8))
        for i in range(wsfuns.shape[0]):
            plt.subplot(6, 7, i+1)
            for j in range(wsfuns.shape[1]):
                plt.plot(wsfuns[i, j, :, :], color = colors[j])
            plt.ylim(ymin, ymax)
        plt.suptitle(f'g = {scales[idx]:.3f}')
        plt.tight_layout()
        plt.show()
        
        plt.savefig(f'anim_frames_scales_{args.task_str}/anim_{idx}.png')

    plt.show()
