from kpflow.tasks import CustomTaskWrapper
from kpflow.architecture import BasicRNN
from kpflow.analysis_utils import ping_dir
import torch, numpy as np, sys, matplotlib.pyplot as plt
from kpflow.architecture import BasicRNN, Model, get_cell_from_model
from itertools import product

sys.path.append('../') # For common code and training code.
from train import parse_arguments, train

args = parse_arguments()

# Enforce some hyperparms (e.g. true GD without other stuff)
args.model = 'rnn'
args.task_str = 'low_rank_forecast'
args.init_level = 1.
args.checkpoint = ''
args.tol = 1e-3
args.niter = int(1e5)
args.lr = 1e-2
args.duration = 10
args.grad_clip = None
args.model_type = 'rnn'
args.save_freq = 5000

def construct_model(g, D_inp, D_out):
    model = Model(input_size = D_inp, output_size = D_out, rnn=args.model_type, hidden_size = 256)

    # Scale inital weights.
    for name, param in model.named_parameters():
        if name == 'rnn.weight_hh_l0':
            param.data = param.data * args.init_level
            print("HIT")

    return model

def construct_task(D_inp, D_out):
    return CustomTaskWrapper('low_rank_forecast', 500, use_noise = True, n_samples = 5000, T = args.duration, seed_data = 1, D_inp = D_inp, D_targ = D_out)

# Setup hard-coded task and models and filenames and send them to train.py
path = f'data_lr={args.lr}/'
ping_dir(path)

g_vals = np.linspace(0., 2.5, 5).tolist()
g_vals = [0., 1., 2.]
Dinp_vals = np.linspace(2, 100, 5).astype(int).tolist()
D_out = 4
models_and_tasks = {
    f'{path}/{args.optim}_rnn_mempro_nfps_g={g}_Dinp={D_inp}': 
        (
            construct_model(g, D_inp, D_out),
            construct_task(D_inp, D_out)
        )
    for g, D_inp in product(g_vals, Dinp_vals)
}
print('Models : ', list(models_and_tasks.keys()))

losses_all = []
for name, (model, task) in models_and_tasks.items():
    print(f'Training {name}...')
    args.save_dir = name
    losses_all.append(train(args, task, model))
losses_all = np.array(losses_all).T
plt.plot(losses_all)
plt.yscale('log')
plt.show()
