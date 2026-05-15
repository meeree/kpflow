from kpflow.tasks import CustomTaskWrapper
from kpflow.analysis_utils import ping_dir, load_checkpoints, import_checkpoint, torch_to_np, np_to_torch, cos_similarity
from kpflow.architecture import Model, get_cell_from_model

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
from tqdm import tqdm

from common import project, plot_trajectories, compute_svs, set_mpl_defaults

def parse_arguments(parser = None):
    parser = argparse.ArgumentParser(description='Plot PCA Dynamics of Trained Model') if parser is None else parser
    parser.add_argument('--model', default='gru', type = str, help='Model to use')
    parser.add_argument('--task_str', default = 'memory_pro', type = str, help = 'Task to train on. Options: memory_pro, flip_flop, context_integration')
    parser.add_argument('--save_dir', default = '', type = str, help = 'Directory where checkpoints were saved. Optional.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    set_mpl_defaults(14)
    PALLETTE = ['#f86969ff', '#7e69f8ff', '#f8c969ff', '#69f87cff', '#e569f8ff']

    task_nice_str = args.task_str.replace('_', ' ').title()
    print(f'Evaluating Results for {task_nice_str}')
    task = CustomTaskWrapper(args.task_str, 100, use_noise = False, n_samples = 100)
    inputs, targets = task()
    n_in, n_out = inputs.shape[-1], targets.shape[-1]

    filename = args.save_dir
    checkpoints, gd_itr = load_checkpoints(filename)
    print(len(checkpoints), filename)
    print(f'Re-Evaluating {len(checkpoints)} Snapshots in {filename}...')
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

    plt.figure()
    plt.plot(out[0, :, 0].detach().cpu().numpy(), c = 'black')
    plt.plot(targets[0, :, 0], linestyle = 'dashed', c = 'black')
    plt.legend(['Output', 'Target'])
    plt.gca().set_prop_cycle(None)
    plt.plot(out[0, :, 1:].detach().cpu().numpy())
    plt.gca().set_prop_cycle(None)
    plt.plot(targets[0, :, 1:], linestyle = 'dashed')
    plt.xlabel('Evaluation Time, $t$')

    plt.figure()
    plt.plot(gd_itr, test_losses)
    plt.title('Test loss')
    plt.xlabel('GD Epoch')
    plt.ylabel('Loss (mse)')

    print(f'Projecting with PCA...')
    pcas, proj_all, proj_w_out_all = [], [], []
    for hidden, model in zip(hidden_all, models):
        pca, proj = project(hidden)
        pcas.append(pca)
        proj_all.append(proj)

        proj_w_out_all.append(model.Wout.weight.data.detach().numpy() @ pca.components_.T) # Project the output rows into the pca space.
    proj_all, proj_w_out_all = np.stack(proj_all), np.stack(proj_w_out_all)

    effdims = []
    for pca in pcas:
        effdims.append(1. / np.sum(pca.explained_variance_ratio_**2)) # Participation ratio.

    plt.figure()
    plt.plot(effdims)
    plt.ylabel('PR Dimension')
    plt.xlabel('GD Epoch')

    def plot_trajectories(data, m = 1, n = 1, i = 1, dim = 3, legend = True, colors = None):
        # data should be shape [batch count, time, hidden dim].
        plt.subplot(m, n, i, projection = None if dim == 2 else '3d')
        for idx, traj in enumerate(data):
            if dim == 3:
                plt.plot(traj[:, 0], traj[:, 1], traj[:, 2])
                plt.gca().set_zlabel('PC3')
            else:
                plt.plot(traj[:, 0], traj[:, 1])
            plt.xlabel('PC1')
            plt.ylabel('PC2')

    plt.figure(figsize = (10, 6))
    plot_trajectories(proj_all[-1])
    plt.title('Model Latent Dynamics')
    plt.show()
