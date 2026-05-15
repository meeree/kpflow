# ###########################################################################################################################################
# Some simple common training code to be used in the examples. 
# To import, first do sys.path.append('../') then something like "from train import train, parse_arguments" 
# You can append arguments to command line by first parsing your own then sending the parser to parse_arguments to append train's args.
# ############################################################################################################################################

from kpflow.tasks import CustomTaskWrapper
from kpflow.analysis_utils import ping_dir, import_checkpoint
from kpflow.architecture import Model, BasicRNN
from torch_optimizer import Shampoo

import torch, argparse, json, numpy as np
from torch import nn
from tqdm import tqdm

def parse_arguments(parser = None):
    parser = argparse.ArgumentParser(description='Common Training Code') if parser is None else parser
    parser.add_argument('--model', default='basic_rnn', type = str, help='Model to use')
    parser.add_argument('--task_str', default = 'memory_pro', type = str, help = 'Task to train on. Options: memory_pro, flip_flop, context_integration')
    parser.add_argument('--save_freq', default=100, type = int, help='Frequency (iterations) to save checkpoints at')
    parser.add_argument('--lr', default = 1e-3, type = float, help = 'Learning rate')
    parser.add_argument('--niter', default = 10000, type = int, help = '# of Iterations for GD')
    parser.add_argument('--init_level', type=float, default=1., help='initialization level for xavier uniform weights')
    parser.add_argument('--save_dir', type=str, default='', help='Directory to save in. Will be set if empty.')
    parser.add_argument('--wandb', type=str, default='', help='Name of wandb project. Is not used if not set.')
    parser.add_argument('--checkpoint', type=str, default='', help='Checkpoint to start from.')
    parser.add_argument('--optim', type=str, default='sgd', help='adam or sgd or adagrad')
    parser.add_argument('--tol', type=float, default=1e-3, help='tolerance for stopping early')
    parser.add_argument('--duration', type=int, default=90, help='task duration')
    parser.add_argument('--grad_clip', type=float, default=None, help='grad clip')
    parser.add_argument('--no_input_noise', action='store_true', help='disable noise in inputs')
    return parser.parse_args()
import torch

@torch.no_grad()
def max_effective_lr_adam(optimizer, eps_floor=1e-16):
    """
    Returns max effective LR over all parameter elements for Adam/AdamW.
    Call after backward() when grads exist. Works best after at least 1 optimizer step
    (so exp_avg/exp_avg_sq are initialized).
    """
    max_eff, min_eff = 0.0, np.inf

    for group in optimizer.param_groups:
        lr = group["lr"]
        eps = group.get("eps", 1e-8)

        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad

            state = optimizer.state.get(p, None)
            if not state or "exp_avg" not in state or "exp_avg_sq" not in state:
                # Before the first step, Adam state isn't populated, so "effective LR" isn't defined yet.
                continue

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            step = state.get("step", 0)

            # bias corrections (match PyTorch Adam)
            beta1, beta2 = group["betas"]
            if step is None or step == 0:
                continue

            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step

            m_hat = exp_avg / bias_correction1
            v_hat = exp_avg_sq / bias_correction2

            denom = v_hat.sqrt().add_(eps)

            # elementwise effective lr: |update| / (|grad| + tiny)
            # update = lr * m_hat / denom
            update = (lr * m_hat) / denom
            eff = update.abs() / (g.abs() + eps_floor)

            # track global max
            max_eff = max(max_eff, float(eff.max().item()))
            min_eff = min(min_eff, float(eff.min().item()))

    return max_eff, min_eff

def effdim(data):
    data_flat = data.reshape((-1, data.shape[-1]))
    mat = data_flat.T @ data_flat / data_flat.shape[0]
    return np.trace(mat)**2 / np.trace(mat @ mat)

def augment(inputs, nfeats = 3):
    freqs = torch.linspace(1., 10., nfeats).to(inputs.device)
    times = 2 * np.pi * torch.linspace(0., 1., inputs.shape[1]).to(inputs.device)
    feats = torch.cos(freqs[None, :] * times[:, None]) # [n_t, nfeats]
    feats2 = torch.zeros_like(feats)
    window = feats.shape[0] // nfeats
    feats3 = []
    for window in [30, 5, 10, 20, 30, 60]:
        inc = feats.shape[0] // window 
        for idx, off in enumerate(range(0, feats.shape[0], inc)):
            f = torch.zeros(feats.shape[0])
            f[off: off +inc] = 1.
            feats3.append(f)

    feats3 = torch.stack(feats3, 1).to(inputs.device)
    feats3 = torch.cat((feats, feats3), -1)

    stim1 = inputs[:, 0, 1]
    feats_batched = stim1[:, None, None] * feats3[None] # [n_x, n_t, nfeats]
    feats_batched = torch.ones_like(stim1[:, None, None]) * feats3[None]
    new_inps = torch.cat([inputs, feats_batched], -1)
    print(new_inps.shape[-1])
    print(effdim(inputs), effdim(new_inps))
    return new_inps

def train(args, task=None, model=None, out_chan=0):
    torch.manual_seed(0)
    np.random.seed(0)
    if task is None:
        task = CustomTaskWrapper(args.task_str, 500, use_noise = not args.no_input_noise, n_samples = 5000, T = args.duration)
    inputs, targets = task()
    n_in_no_aug = int(inputs.shape[-1])
#    inputs = augment(inputs)
    n_aug = inputs.shape[-1] - n_in_no_aug
    n_in, n_out = inputs.shape[-1], targets.shape[-1]

#    Win = model.rnn.weight_ih_l0
#    Waug_data = torch.rand(Win.shape[0], n_aug)
#    Waug_data *= (Win.norm() / Waug_data.norm())
#    Win.data = torch.cat((Win.data, Waug_data.to(Win.device)), -1)

    losses_all = []

    device = 'cuda'

    # Initialize model and move to appropriate device
    model_type = {'gru': nn.GRU, 'rnn': nn.RNN, 'basic_rnn': BasicRNN}[args.model]
    if model is None:
        model = Model(input_size = n_in, output_size = n_out, rnn=model_type, hidden_size = 256)

#    hidden = model(inputs)[1]
#    V = torch.cat((hidden, inputs), -1)
#    print('V dim', effdim(V.detach()))
#    ajsoidsadj

    # Scale inital weights.
    for name, param in model.named_parameters():
        if name == 'rnn.weight_hh_l0':
            param.data = param.data * args.init_level
#    model.rnn.flatten_parameters()

    optim_type = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD, 'adagrad': torch.optim.Adagrad, 'shampoo': Shampoo}[args.optim.lower()]
    if optim_type == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr = args.lr, eps = 1e-6, betas = (0.9, 0.999))
    else:
        optim = optim_type(model.parameters(), lr = args.lr) 
    loss_fn = nn.MSELoss()
    losses = []

    if args.wandb != '':
        import wandb
        wandb_config = vars(args)
        wandb.init(project = args.wandb, config = wandb_config)


    model = model.to(device)
    path = f'{args.task_str}_{args.model}_init={args.init_level}/' if not args.save_dir else args.save_dir
    if path[-1] != '/':
        path = path + '/'

    if args.checkpoint != '':
        model.load_state_dict(import_checkpoint(args.checkpoint)['model'])

    ping_dir(path)

    # Save the configurationm.
    with open(f'{path}config.json', 'w') as fl:
        json.dump(vars(args), fl, indent = 3)

    ping_dir(f'{path}checkpoints/', clear = True)

    checkpoints = []
    pbar = tqdm(range(args.niter))
    start_loss = None
    for itr in pbar:
        inputs, targets = task()
        inputs, targets = inputs.to(device), targets.to(device)
#        inputs = augment(inputs)

        optim.zero_grad()
        out = model(inputs)[out_chan] if out_chan is not None else model(inputs) # No output channel, use whole thing.

        loss = loss_fn(out, targets)
        loss.backward()
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if start_loss is None:
            start_loss = loss.item()

        pbar.set_description(f'Loss {loss:.2e}; % from init loss to tol {100*(start_loss - loss) / (start_loss - args.tol):f};')

        done = itr == args.niter - 1
        if len(losses) > 2:
            if (losses[-1] + loss.item()) / 2. < args.tol:
                done = True # Converged

        if itr % 20 == 0:
            losses.append(loss.item())
            if args.wandb != '':
                max_lr, min_lr = max_effective_lr_adam(optim)
                grad_sq = sum((p.grad**2).sum() for p in model.parameters())
#                Win_aug = model.rnn.weight_ih_l0
#                Win, Waug = Win_aug[:, :-n_aug], Win_aug[:, -n_aug:]
                log_entry = {"loss": losses[-1], 'grad_norm': grad_sq ** .5, 'max_lr': max_lr, 'min_lr': min_lr}#, 'Win_norm': Win.data.norm(), 'Waug_norm': Waug.data.norm()}
                wandb.log(log_entry)

        if done or itr % args.save_freq == 0:
            snapshot = {
                'model' : model.state_dict(),
                'optim' : optim.state_dict(), 
                'init_lr' : args.lr,
                'model_type' : args.model,
                'iteration' : itr
            }
            torch.save(snapshot, f'{path}checkpoints/checkpoint_{itr}.pt')

#        if itr % 1000 == 0:
#            N = len(list(model.parameters()))
#            H = hessian(loss_fn)(list(model.parameters()))
#            H_matrix = torch.cat([h.flatten() for h in H]).reshape(N,N)
#
#            eigs = torch.linalg.eigvals(H_matrix).real
#            print(f'Hessian eigs min max: {eigs.min():.3e}, {eigs.max():.3e}')

        optim.step()

        if done:
            break

    if args.wandb != '':
        wandb.finish()

    return losses

if __name__ == '__main__':
    train(parse_arguments())
