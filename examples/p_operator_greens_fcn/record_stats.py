import matplotlib.pyplot as plt
import numpy as np
import torch, os, glob
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tqdm import tqdm
from itertools import product
from train import GetTask, GetModel
from kpflow.grad_op import HiddenNTKOperator as NTK

def GetTaskUniform(nsamp): # nsamp should be a square.
    width = np.sqrt(2 * np.pi) # So area of +1 and -1 regions are same.
    nsamp_x = int(nsamp ** 0.5)
    X = torch.linspace(-width/2, width/2, nsamp_x)
    inps = torch.stack(torch.meshgrid(X, X)).reshape((2, nsamp)).T
    targs = torch.where(torch.linalg.norm(inps, axis = 1) > 1, -1, 1)[:, None].float()
    return inps, targs

class EvalAll(nn.Module): # Evaluate all activations of model and output it, not traditional output.
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        states = []
        for layer in self.net:
            states.append(layer(x))
            x = states[-1]
        return torch.stack(states[:-1]) # Hidden activations.

nsamp_x = 50
nsamp = nsamp_x**2
to_grid = lambda X: X.reshape((nsamp_x, nsamp_x))
to_flat = lambda X: X.reshape((nsamp))

inps, targs = GetTaskUniform(nsamp)

files = glob.glob('data/sweep/model*.pt')
print('Computing analysis statistics and saving them to data files.')
for fl in tqdm(files):
    ld = torch.load(fl)
#    plt.imshow(ld['ftle'].reshape((50,50)))
#    plt.show()
    config, sd = ld['config'], ld['model']
    L, N, gain, seed = config['depth'], config['width'], config['gain'], config['seed']

    model = GetModel(L, N, gain, True)
    model.load_state_dict(sd)

    # Compute the FTLE for the transfer of information from the task input to the final hidden unit activation.
    if 'ftle' not in ld.keys():
        model_hidd_sum = lambda x : model[:-1](x).sum(0) # Map inputs to final hidden and sum over batches.

        # Note that jac is of shape (nsamp, 100, 2). So it makes no sense to use some complicated jvp matrix-free shit here. Each matrix only has two singular vals!
        jac = torch.autograd.functional.jacobian(model_hidd_sum, inps).moveaxis(1,0)
        lyap = torch.log(torch.linalg.svdvals(jac)) / (L-1)
        ftle = lyap[:, 0]

        # Re-save the data but with new analysis stuff appended.
        ld['ftle'] = ftle.detach()

    torch.manual_seed(seed)
    model_0 = GetModel(L, N, gain, True)

    model_0_all, model_f_all = EvalAll(model_0), EvalAll(model) # Evaluate all layer activations (not just final)
    hidd_0, hidd_f = model_0_all(inps), model_f_all(inps) 

    # NTK Alignment.
    if 'kern_align' not in ld.keys():
        ntk_0, ntk_f = NTK(model_0_all, inps, hidd_0), NTK(model_f_all, inps, hidd_f)
        kern_align = ntk_f.alignment(ntk_0, nsamp = 50)
        ld['kern_align'] = kern_align

    # Representation Alignment.
    if 'rep_align' not in ld.keys():
        H0, Hf = hidd_0.flatten(), hidd_f.flatten()
        ld['rep_align'] = (torch.inner(H0, Hf) / (torch.inner(H0, H0) * torch.inner(Hf, Hf))**0.5).item()

    torch.save(ld, fl)

plt.imshow(to_grid(ftle), cmap = 'seismic')
plt.colorbar()
plt.tight_layout()
plt.show()
