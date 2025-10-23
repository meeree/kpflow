from kpflow.tasks import CustomTaskWrapper
from kpflow.analysis_utils import ping_dir, load_checkpoints, import_checkpoint, torch_to_np, np_to_torch, cos_similarity
from kpflow.architecture import Model, get_cell_from_model
from kpflow.parameter_op import ParameterOperator, JThetaOperator
from kpflow.propagation_op import PropagationOperator_DirectForm, PropagationOperator_LinearForm
from kpflow.grad_op import HiddenNTKOperator
from kpflow.op_common import AveragedOperator, Operator
from kpflow.trace_estimation import op_alignment

import torch
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
    parser.add_argument('--file_1', type = str)
    parser.add_argument('--file_2', type = str)
    parser.add_argument('--file_3', type = str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    set_mpl_defaults(14)
    PALLETTE = ['#f86969ff', '#7e69f8ff', '#f8c969ff', '#69f87cff', '#e569f8ff']

    task_nice_str = args.task_str.replace('_', ' ').title()
    print(f'Evaluating Results for {task_nice_str}')

    task = CustomTaskWrapper(args.task_str, 20, use_noise = False, n_samples = 20, T = 90)
    inputs, targets = task()
    n_in, n_out = inputs.shape[-1], targets.shape[-1]

    plt.figure(figsize = (12, 3))
    for file_idx, filename in enumerate([args.file_1, args.file_2, args.file_3]):
        checkpoints, gd_itr = load_checkpoints(filename)
        print(len(checkpoints), filename)
        print(f'Re-Evaluating {len(checkpoints)} Snapshots in {filename}...')

        g_val = float(re.search(r"init_([0-9.]+)", filename).group(1)) # Scale should be after init_ in the filename. 
        test_losses, models, hidden_all = [], [], []
        for ch in checkpoints:
            model = Model(input_size = n_in, output_size = n_out, rnn=nn.GRU if args.model == 'gru' else nn.RNN, hidden_size = 256)
            model.load_state_dict(import_checkpoint(ch)['model'])
            out, hidden = model(inputs)
            test_losses.append(nn.MSELoss()(out, targets).item())
            hidden_all.append(torch_to_np(hidden))
            models.append(model)
        hidden_all = np.stack(hidden_all)
        print(f'Hidden shape over all GD snapshots has shape {hidden_all.shape} = (GD Iter, Trial, Time, Hidden Unit)')

        dims = [[], [], []]
        for idx, (model, hidden) in enumerate(zip(tqdm(models[::4]), hidden_all[::4])):
            cell = get_cell_from_model(model)
            pop = PropagationOperator_LinearForm(cell, inputs, hidden)
            kop = ParameterOperator(cell, inputs, hidden)

            class GetHidden(nn.Module):
                def __init__(self, net):
                    super().__init__()
                    self.net = net

                def forward(self, x):
                    return self.net(x)[1]

            gop = HiddenNTKOperator(GetHidden(model), inputs, hidden)

            dims[0].append(op_alignment(gop, pop @ pop.T()))
            dims[1].append(0)
            dims[2].append(0)
            continue

            avg_shape = (1, 1, hidden.shape[2])
            for idx2, op in enumerate([kop, pop, gop]):
                nsamp = 100
                S = torch.randint(2, size=(nsamp,*hidden.shape)).float() * 2 - 1 # Either 1 or -1
                Y = op.batched_call(S)
                Y = Y.reshape((-1, Y.shape[-1])).detach().cpu().numpy()
                C_emp = Y.T @ Y / nsamp
                svals, svecs = np.linalg.eigh(C_emp)
                svals, svecs = svals[::-1], svecs[:, ::-1]
                dims[idx2][idx].append(np.argmax(np.cumsum(svals**2 / (svals**2).sum()) > .95) + 1)
                continue
                
#                Y_m = Y.reshape((nsamp, -1, Y.shape[-1]))
#                M = np.sum(Y_m * Y_m, axis = 1)
#                M = Y_m.swapaxes(1, 2) @ Y_m.swapaxes(0, 1)
#                C_emp_running = np.cumsum(M, axis = 0) / np.arange(1, nsamp+1)[:, 0] # Running mean.
#                var = np.cumsum((M - C_emp_running)**2, axis = 0) / (np.arange(1, nsamp+1) * np.arange(0, nsamp))[:,0] 
#                plt.plot(np.sum(var, 


                plt.show()
                askdoisajdsa

                ncomp = 255 if idx == 1 else 80
                ncomp = 20
                sv, sfuns = op.svd(ncomp, compute_vecs = True)
                wsfuns = sv[:, None, None, None] * sfuns
                print(sfuns.shape)
                wsfuns_flat = wsfuns.reshape((-1, wsfuns.shape[-1]))
                print(wsfuns_flat.shape)
                pca = PCA().fit(wsfuns_flat)
                print(np.argmax(np.cumsum(pca.explained_variance_ratio_) > .95) + 1)
                aoisjdoisajd


                sv = op.svd(ncomp, avg_shape)
                dims[idx].append(Operator.effrank(sv, .95)[0])

        plt.subplot(1, 3, 1 + file_idx)
        plt.plot(dims[0], color = PALLETTE[0], linewidth = 3)
        plt.plot(dims[1], color = PALLETTE[1], linewidth = 3)
        plt.plot(dims[2], color = PALLETTE[2], linewidth = 3)
        plt.title(f'Init. Scale g = {g_val:.1f}')

        if file_idx == 0:
            plt.legend(['$\\mathcal{K}$', '$\\mathcal{P}$', '$\\mathcal{P K P^*}$'])
            plt.xlabel('GD Iteration')
            plt.ylabel('Operator Effective Rank')

        plt.tight_layout()

    plt.savefig(f'Effrank_over_gd_{args.task_str}.png')
    plt.show()
