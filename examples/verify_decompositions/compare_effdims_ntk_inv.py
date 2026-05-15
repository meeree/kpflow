import numpy as np
import matplotlib.pyplot as plt
import torch
from kpflow.tasks import CustomTaskWrapper
from torch import nn
from kpflow.architecture import BasicRNNCell, Model, get_cell_from_model
from kpflow.analysis_utils import ping_dir, load_checkpoints, import_checkpoint, torch_to_np, np_to_torch, cos_similarity
from tqdm import tqdm
from kpflow.grad_op import HiddenNTKOperator
from kpflow.op_common import Operator, MatrixWrapper, IdentityOperator
from kpflow.parameter_op import ParameterOperator, JThetaOperator
from kpflow.propagation_op import PropagationOperator_DirectForm, PropagationOperator_LinearForm

import sys
sys.path.append('../')
from common import project, plot_trajectories, compute_svs, set_mpl_defaults, plot_traj_mempro, imshow_nonuniform, effdim, relative_error, plot_err_bar, skree_plot

class GetHidden(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)[1]

#task = CustomTaskWrapper('low_rank_forecast', 20, use_noise = False, n_samples = 20, T = 30, seed_data = 10)#, D_inp = 10, D_targ = 10)
task = CustomTaskWrapper('memory_pro', 20, use_noise = False, n_samples = 20, T = 90)

inputs, targets = task()
n_in, n_out = inputs.shape[-1], targets.shape[-1]

dt = 1
checkpoints = load_checkpoints('../simplicity_bias/memory_pro_basic_rnn_init=0/')[0]
#checkpoints = load_checkpoints('../simplicity_bias/memory_pro_basic_rnn_init=1.0/')[0]
model = Model(n_in, 256, n_out, rnn = BasicRNNCell, bias = True, linear = True, dt = dt)
#model.rnn.weight_hh_l0.data *= 0.5
model.load_state_dict(import_checkpoint(checkpoints[0])['model'])

out, hidden = model(inputs)

n_x, n_t, n = hidden.shape

if False:
    # Plot evolution of resolvent eigenvalues over GD.
    for idx, checkpoint in enumerate(checkpoints):
        sd = import_checkpoint(checkpoint)['model']
        W = sd['rnn.weight_hh_l0']
        R = torch.linalg.inv(torch.eye(n) - W)
        Lam = torch.linalg.svdvals(R)
        print(checkpoint)
        plt.plot(Lam)
    plt.show()

bipart = lambda x : x.reshape((-1, x.shape[-1])) # Partition the full 3-tensor space into a time-trials part and a physical part.
W = model.rnn.weight_hh_l0.data # Weights of the model.
jac = (1-dt) * torch.eye(n) + dt * W.clone()
W = jac # Use true Jacobian, not W, when using continuous time.
hidden_shift = torch.cat((0. * hidden[:, :1], hidden), 1)[:, :-1]
V = torch.cat((bipart(hidden_shift), bipart(inputs)), -1)
err = bipart((targets - out) @ model.Wout.weight)

R = torch.linalg.inv(torch.eye(n) - W)
Q, Lam, _ = torch.linalg.svd(R, full_matrices = False)
Lam2 = Lam**2
Lam_varrat = Lam2 / Lam2.sum()

B, Sig, _ = torch.linalg.svd(V, full_matrices = False)
Sig2 = Sig**2
Sig_varrat = Sig2 / Sig2.sum()

B_pinv = torch.linalg.pinv(B)
Q_pinv = torch.linalg.pinv(Q)

ntk = HiddenNTKOperator(GetHidden(model), inputs, hidden)

B_out, Q_out = B.shape[0], Q.shape[0]
B_in, Q_in = B.shape[1], Q.shape[1]
nat_to_euc_op = MatrixWrapper(B).tprod(MatrixWrapper(Q)).reshape((B_in, Q_in), ntk.shape_in)  # Natural basis to Euclidean basis.
euc_to_nat_op = nat_to_euc_op.T # Since the matrices B and Q are orthogonal.

if False:
    # Project NTK onto different time periods.
    P_final_times = torch.eye(n_t)
    P_final_times[:30, :30] = 0.
    P_final_op = IdentityOperator(n_x).tprod(MatrixWrapper(P_final_times)).tprod_like(IdentityOperator(n), ntk)
    P_init_times = torch.eye(n_t)
    P_init_times[30:, 30:] = 0.
    P_init_op = IdentityOperator(n_x).tprod(MatrixWrapper(P_init_times)).tprod_like(IdentityOperator(n), ntk)
    ntk_final_times = P_final_op @ ntk
    ntk_init_times = P_init_op @ ntk
    print(ntk_final_times.fro_norm(30)**2 / ntk.fro_norm(30)**2, ntk_init_times.fro_norm(30)**2 / ntk.fro_norm(30) ** 2)

if True:
    # Measure if NTK is diagonalizable by the B and Q bases.
    ntk_nat = euc_to_nat_op @ ntk @ nat_to_euc_op
    ntk_surr = MatrixWrapper(V @ V.T).tprod_like(MatrixWrapper(R @ R.T), ntk)
    ntk_surr_nat = euc_to_nat_op @ ntk_surr @ nat_to_euc_op

    diag_op = MatrixWrapper(torch.diag(Sig2)).tprod(MatrixWrapper(torch.diag(Lam2)))

    print(ntk_nat.alignment(diag_op, nsamp = 20))
    print(ntk_surr_nat.alignment(diag_op, nsamp = 20))

    plt.plot(torch.diag(ntk_nat.partial_trace(0).full_matrix()).detach())
    plt.gca().twinx()
    plt.plot((torch.linalg.norm(Sig) * Lam2).detach(), color = 'red')
    plt.show()

proj_natural = lambda X: (B.T @ X @ Q).detach()

V_proj = proj_natural(bipart(hidden_shift))

energy = Sig2[:,None] * Lam2[None,:]
energy = energy / energy.sum()

if False:
    # Plot time modes.
    nmode = 3
    plt.figure()
    for idx in range(nmode):
        plt.subplot(nmode,1,1+idx)
        plt.plot(B.reshape((n_x, n_t, -1))[:, :, idx].T.detach())
        plt.title(f'Mode $b_{idx+1}$')
        plt.ylim(B[:,:nmode].min().item()*1.1, B[:,:nmode].max().item()*1.1)

    plt.xlabel('Time, $t$')
    plt.show()

plt.subplot(1,2,1)
#plt.plot(torch.cumsum(Sig_varrat,0).detach())
wout = model.Wout.weight[:1, :]
targets_toy = inputs[:, :, 1:2].clone()
shift = 0
targets_toy = torch.cat((0.*targets_toy[:, :shift], targets_toy), 1)[:, :n_t]
err_toy = bipart((targets_toy - out[:, :, 1:2]) @ wout)
err_proj = proj_natural(bipart(err_toy))
err_varrat = err_proj**2 / (err_proj**2).sum()
skree_mat_err = torch.cumsum(torch.cumsum(err_varrat, 0), 1).detach()
plt.imshow(skree_mat_err, vmin = 0, vmax = 1, origin = 'lower')
plt.xlabel('Space Modes')
plt.ylabel('Time-Trial Modes')

#plt.plot(V_proj)
#plt.plot(Sig.detach())
plt.subplot(1,2,2)
skree_mat = torch.cumsum(torch.cumsum(energy, 0), 1).detach()
plt.imshow(skree_mat, vmin = 0, vmax = 1, origin = 'lower')
#err_proj = proj_natural(bipart(hidden_shift))
#plt.plot(err_proj)
#plt.plot(Sig.detach())
plt.xlabel('Space Modes')

plt.figure()
plt.subplot(1,2,1)
plt.plot(skree_mat[-1, :])
plt.plot(skree_mat_err[-1, :])

plt.subplot(1,2,2)
plt.plot(skree_mat[:, -1])
plt.plot(skree_mat_err[:, -1])

wout = model.Wout.weight[:1, :]
targets_toy = inputs[:, :, 1:2].clone()
skree_mat = torch.cumsum(torch.cumsum(energy, 0), 1).detach()

plt.figure()
#plt.subplot(1,2,1)
#plt.plot(skree_mat[-1, :])
#plt.subplot(1,2,2)
plt.plot(skree_mat[:, -1])
for shift in range(0, 70, 10):
    targets_toy = torch.cat((0.*targets_toy[:, :shift], targets_toy), 1)[:, :n_t]
    err_toy = bipart((targets_toy - out[:, :, 1:2]) @ wout)
    err_toy = bipart(targets_toy @ wout)
    err_proj = proj_natural(err_toy)
    err_varrat = err_proj**2 / (err_proj**2).sum()
    skree_mat_err = torch.cumsum(torch.cumsum(err_varrat, 0), 1).detach()

#    plt.subplot(1,2,1)
#    plt.plot(skree_mat_err[-1, :], label = shift)
#    plt.subplot(1,2,2)
    print(skree_mat_err[:, -1])
    plt.plot(skree_mat_err[:, -1], label = shift)

plt.xlabel('Number of Temporal Modes')
plt.ylabel('Cumulative Variance Ratio')
plt.legend()

plt.show()

def is_inverse(A, A_inv_guess):
    Id = IdentityOperator(A.shape_in)
    return (A @ A_inv_guess).alignment(Id)

def pinv_check_1(A, A_pinv):
    return (A @ A_pinv @ A).alignment(A)

set_mpl_defaults(14)
colors = [*plt.rcParams['axes.prop_cycle'].by_key()['color']]

n = 100
W = np.random.randn(n, n) / np.sqrt(n)
gs = np.linspace(1e-10, 3., 30)

alignment_guess = []
for g in tqdm(gs):
    model = Model(n_in, 256, n_out, rnn = BasicRNNCell, bias = False, linear = True)
    model.rnn.weight_hh_l0.data *= g
    hidden = model(inputs)[1]
    n_x, n_t, n = hidden.shape
    bipart = lambda x : x.reshape((-1, x.shape[-1])) # Partition the full 3-tensor space into a time-trials part and a physical part.

    class GetHidden(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x)[1]

    dt = 0.01
    ntk = HiddenNTKOperator(GetHidden(model), inputs, hidden)
    hidden_shift = torch.cat((0. * hidden[:, :1], hidden), 1)[:, :-1]
    V = torch.cat((bipart(hidden_shift), bipart(inputs)), -1)
    G = torch.linalg.pinv(V @ V.T)
    G_op = MatrixWrapper(G)
    W = model.rnn.weight_hh_l0.data # Weights of the model.
    jac = (1-dt) * torch.eye(n) + dt * W.clone()
    W = jac
    W_op = MatrixWrapper(W)
    I_W_op = MatrixWrapper(torch.eye(n) - W)

    cell = get_cell_from_model(model)
    pop = PropagationOperator_LinearForm(cell, inputs, hidden)
    kop = ParameterOperator(cell, inputs, hidden)

    Dt = torch.eye(n_t)
    Dt[range(1,n_t), range(n_t-1)] = -1.
    Dt_op = IdentityOperator(n_x).tprod(MatrixWrapper(Dt)).like(G_op)

#    plt.imshow(((V @ V.T) @ torch.linalg.pinv(V @ V.T)).detach())
#    plt.show()

    guess_1 = G_op.tprod_like(I_W_op.T.gram(), ntk)
    guess_2 = guess_1 + (G_op @ Dt_op).tprod_like(I_W_op.T @ W_op, ntk).symm_part() * 2.
    guess_3 = guess_2 + (Dt_op.T @ G_op @ Dt_op).tprod_like(W_op.T.gram(), ntk)
    alignment_guess.append([
        pinv_check_1(ntk, guess_1),
    #    pinv_check_1(ntk, guess_2),
        pinv_check_1(ntk, guess_3),

        pinv_check_1(pop, (IdentityOperator(n_x*n_t).tprod(I_W_op) + Dt_op.tprod(W_op)).like(ntk)),
#        kop.alignment(MatrixWrapper(V @ V.T).tprod_like(IdentityOperator(n), ntk)),
        pinv_check_1(G_op.tprod_like(IdentityOperator(n), ntk), kop)
    ])

alignment_guess = np.stack(alignment_guess)

plt.subplot(1,2,1)
plt.plot(gs, alignment_guess[:, :3])
plt.legend(['$$A_0 \otimes R^{-1}$$', '$$A_0 \otimes R^{-1} + A_1 \otimes I_n$$'])

plt.subplot(1,2,2)
plt.plot(gs, alignment_guess[:, 3:])
plt.legend(['$\mathcal{P}^{-1}$ Guess', '$\mathcal{K}^{-1}$ Guess'])
plt.show()
