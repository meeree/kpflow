from kpflow.tasks import CustomTaskWrapper
from kpflow.analysis_utils import ping_dir, import_checkpoint
from kpflow.architecture import Model, get_cell_from_model

from common import set_mpl_defaults, plot_trajectories, project

import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import argparse 

def parse_arguments(parser = None):
    parser = argparse.ArgumentParser(description='Train Model on Memory Pro Task') if parser is None else parser
    parser.add_argument('--model', default='gru', type = str, help='Model to use')
    parser.add_argument('--task_str', default = 'memory_pro', type = str, help = 'Task to train on. Options: memory_pro, flip_flop, context_integration')
    parser.add_argument('--save_freq', default=100, type = int, help='Frequency (iterations) to save checkpoints at')
    parser.add_argument('--lr', default = 1e-1, type = float, help = 'Learning rate')
    parser.add_argument('--niter', default = 1000, type = int, help = '# of Iterations for GD')
    parser.add_argument('--init_level', type=float, default=0, help='initialization level for xavier uniform weights')
    parser.add_argument('--save_dir', type=str, default='', help='Directory to save in. Will be set if empty.')
    parser.add_argument('--checkpoint', type=str, default='', help='Checkpoint to start from.')
    parser.add_argument('--optim', type=str, default='adam', help='adam or sgd or adagrad')
    parser.add_argument('--tol', type=float, default=1e-3, help='tolerance for stopping early')
    return parser.parse_args()

def main(args):
    set_mpl_defaults(14)
    PALLETTE = ['#f86969ff', '#7e69f8ff', '#f8c969ff', '#69f87cff', '#e569f8ff']

    task = CustomTaskWrapper(args.task_str, 5000, use_noise = True, n_samples = 5000, T = 90)
    inputs, targets = task()
    n_in, n_out = inputs.shape[-1], targets.shape[-1]

    losses_all = []

    device = 'cpu'

    # Initialize model and move to appropriate device
    model = Model(input_size = n_in, output_size = n_out, rnn=nn.GRU if args.model == 'gru' else nn.RNN, hidden_size = 256).to(device)

    # Scale inital weights.
    for name, param in model.named_parameters():
        if name == 'rnn.weight_hh_l0':
            param.data = param.data * args.init_level
    model.rnn.flatten_parameters()

    loss_fn = nn.MSELoss()
    losses = []

    path = f'functiona_gd_{args.task_str}_{args.model}_init={args.init_level}/' if not args.save_dir else args.save_dir
    if args.checkpoint != '':
        model.load_state_dict(import_checkpoint(args.checkpoint)['model'])

    ping_dir(path)
    ping_dir(f'{path}checkpoints/', clear = True)

    checkpoints = []
    pbar = tqdm(range(args.niter))
    start_loss = None

    inputs, targets = task()
    inputs, targets = inputs.to(device), targets.to(device)

#    W = model.Wout.weight.data.detach().numpy()
#    b = model.Wout.bias.data.detach().numpy()
#    y_star = targets.detach().numpy()
#    guess = (-b + y_star) @ np.linalg.inv(W @ W.T) @ W
#    plt.plot(y_star[0, :, :])
#    plt.show()
#    plot_trajectories(guess[:30]) 
#    plt.show()

    def loss_fn_full(hidden):
        return loss_fn(model.Wout(hidden), targets)

    z = model(inputs)[1]

    proj_all = []
    for itr in pbar:
        l = loss_fn_full(z)
        a = torch.autograd.grad(l, z)[0]
        z = z.clone() - args.lr * a

        if itr % 10 == 0:
            pca, proj = project(z.detach().cpu().numpy())
            proj_all.append(proj)

        pbar.set_description(f'Loss {l.item():.2e};')
        losses.append(l.item())

    cell = get_cell_from_model(model)
    pbar = tqdm(range(20))
    optim = torch.optim.Adam(cell.parameters(), lr = 1e-4)
    second_losses = []
    inputs_flat, z_flat = inputs[:,1:].reshape((-1, inputs.shape[-1])), z.detach()[:,:-1].reshape((-1, z.shape[-1]))
    z_next_flat = z.detach()[:, 1:].reshape((-1, z.shape[-1]))
    for epoch in pbar:
        for itr in range(0, inputs_flat.shape[0], 500):
            optim.zero_grad()
            z_new = cell(inputs_flat[itr:itr+500], z_flat[itr:itr+500])
            loss = loss_fn(z_new, z_next_flat[itr:itr+500])
            pbar.set_description(f'Loss {loss.item():.2e};')
            loss.backward()
            optim.step()
            second_losses.append(loss.item())
#
#        idx = torch.randperm(inputs_flat.shape[0])
#        inputs_flat, z_flat, z_next_flat = inputs_flat[idx], z_flat[idx], z_next_flat[idx] # Shuffle

    plt.plot(second_losses)

    zs = [torch.zeros_like(z[:, 0])]
    for t in range(inputs.shape[1]):
        zs.append(cell(inputs[:, t], zs[-1]))

    zs = torch.stack(zs, 1)
    plt.figure()
    pca, proj = project(zs.detach().cpu().numpy())
    plot_trajectories(proj[:50])
    plt.show()


    plt.figure()
    plt.plot(losses)

    # Plot PCA projection
    pca, proj = project(z.detach().cpu().numpy())

    print(1 + np.argmax(np.cumsum(pca.explained_variance_ratio_) > 0.95))

    plt.figure(figsize = (8 * len(proj_all), 6))
    for idx, proj in enumerate(proj_all):
        plot_trajectories(proj[:50], 1, 5, idx + 1)

    plt.title('PCA Projection')
    plt.gca().set_xlabel('PC1')
    plt.gca().set_ylabel('PC2')
    plt.gca().set_zlabel('PC3')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main(parse_arguments())
