from kpflow.tasks import CustomTaskWrapper
from kpflow.architecture import BasicRNN
from kpflow.analysis_utils import ping_dir
from at_init_fig import construct_model
import torch, numpy as np, sys, matplotlib.pyplot as plt

sys.path.append('../') # For common code and training code.
from train import parse_arguments, train

args = parse_arguments()

# Enforce some hyperparms (e.g. true GD without other stuff)
args.model = 'rnn'
args.task_str = 'memory_pro'
args.init_level = 1.
args.checkpoint = ''
args.tol = 1e-3
args.duration = 90
args.grad_clip = None

# Weight scale
g = 1.

# Setup hard-coded task and models and filenames and send them to train.py
path = f'data_grad_clip_lr={args.lr}/'
ping_dir(path)
task = CustomTaskWrapper('memory_pro', 500, use_noise = True, n_samples = 5000, T = 90)
models = {
    f'{path}/{args.optim}_rnn_mempro_nfps={nfps}_g={g}': 
        construct_model(g = g, fps = [torch.randn(size = (256,)) for _ in range(nfps)], dt = .7)
    for nfps in [0, 5] 
}
print('Models : ', list(models.keys()))

losses_all = []
for name, model in models.items():
    print(f'Training {name}...')
    args.save_dir = name
    losses_all.append(train(args, task, model))
losses_all = np.array(losses_all).T
plt.plot(losses_all)
plt.yscale('log')
plt.show()
