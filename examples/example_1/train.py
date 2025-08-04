from kpflow.tasks import CustomTaskWrapper
from kpflow.analysis_utils import ping_dir
from kpflow.architecture import Model

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
    parser.add_argument('--lr', default = 1e-3, type = float, help = 'Learning rate')
    parser.add_argument('--niter', default = 10000, type = float, help = '# of Iterations for GD')
    parser.add_argument('--init_level', type=float, default=0, help='initialization level for xavier uniform weights')
    parser.add_argument('--save_dir', type=str, default='', help='Directory to save in. Will be set if empty.')
    return parser.parse_args()

def main(args):
    task = CustomTaskWrapper(args.task_str, 200, use_noise = True, n_samples = 5000, T = 90)
    inputs, targets = task()
    n_in, n_out = inputs.shape[-1], targets.shape[-1]

    losses_all = []

    device = 'cuda'

    # Initialize model and move to appropriate device
    model = Model(input_size = n_in, output_size = n_out, rnn=nn.GRU if args.model == 'gru' else nn.RNN, hidden_size = 256).to(device)

    # Scale inital weights.
    for name, param in model.named_parameters():
        if name == 'rnn.weight_hh_l0':
            param.data = param.data * args.init_level
    model.rnn.flatten_parameters()

    optim = torch.optim.Adam(model.parameters(), lr = args.lr) 
    loss_fn = nn.MSELoss()
    losses = []

    path = f'{args.task_str}_{args.model}_init={args.init_level}/' if not args.save_dir else args.save_dir
    ping_dir(path)
    ping_dir(f'{path}checkpoints/', clear = True)

    checkpoints = []
    pbar = tqdm(range(args.niter))
    start_loss = None
    for itr in pbar:
        inputs, targets = task()
        inputs, targets = inputs.to(device), targets.to(device)

        optim.zero_grad()
        out = model(inputs)[0]
        loss = loss_fn(out, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .0001)
        optim.step()

        if start_loss is None:
            start_loss = loss.item()

        pbar.set_description(f'Loss {loss:.2e}; Initial Loss {start_loss:.2e}')

        done = itr == 10000 - 1
        if len(losses) > 2:
            if (losses[-1] + loss.item()) / 2. < 1e-3:
                done = True # Converged

        if itr % 20 == 0:
            losses.append(loss.item())

        if done or itr % args.save_freq == 0:
            snapshot = {
                'model' : model.state_dict(),
                'optim' : optim.state_dict(), 
                'init_lr' : args.lr,
                'model_type' : args.model,
                'iteration' : itr
            }
            torch.save(snapshot, f'{path}checkpoints/checkpoint_{itr}.pt')

        if done:
            break

if __name__ == '__main__':
    main(parse_arguments())
