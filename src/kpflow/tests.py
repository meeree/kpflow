import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

np_to_torch = lambda x: x if torch.is_tensor(x) else torch.from_numpy(x)
torch_to_np = lambda x: x if not torch.is_tensor(x) else x.detach().cpu().numpy()

def check_adjoint(A, trials=5, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    m, n = A.shape
    rel_err = []
    for i in range(trials):
        x = rng.standard_normal(n) + 1j*rng.standard_normal(n) if np.iscomplexobj(A.matvec(np.ones(n))) else rng.standard_normal(n)
        y = rng.standard_normal(m) + 1j*rng.standard_normal(m) if np.iscomplexobj(A.matvec(np.ones(n))) else rng.standard_normal(m)
        lhs = np.vdot(A.matvec(x), y)      # <Ax, y>
        rhs = np.vdot(x, A.rmatvec(y))     # <x, A* y>
        rel_err.append(abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1))
    return np.stack(rel_err)

def absolute_error(x, y):
    return np.abs(torch_to_np(x) - torch_to_np(y)).max()

def relative_error(x, y):
    return absolute_error(x, y) /  max(np.abs(torch_to_np(x)).max(), np.abs(torch_to_np(y)).max())

def test_linearized_propagation(plot = False):
    # First, test on a linear ODE model, z_{n+1} = W * z_n + x(t). 
    # This has dynamics z_n = Phi(n, 0) z_0 + sum_{i=1}^n Phi(n, i) x_i,
    # where Phi(n, n0) = W^{n - n0}.
    from kpflow.propagation_op import PropagationOperator_LinearForm as POLF

    B, T, H = 5, 20, 10 # Batch size, Timesteps, Hidden count.
    x = torch.randn(B, T, H) * 1e-3 # Inputs.
    W = torch.randn(10, 10) / (H**0.5)
    model_f = lambda x, z: z @ W.T + x 

    # Simulate an example.
    hidden = [torch.zeros((B, H))]
    for t in range(T):
        hidden.append(model_f(x[:, t], hidden[-1]).clone())
    hidden = torch.stack(hidden[1:], 1) # [B, T, H]

    if plot:
        plt.plot(hidden[0, :, :])
        plt.show()

    polf = POLF(model_f, 0.*x, hidden)

    print(" ----- ")
    print("Jacobian Matrices Relative Error:")
    true_jac = torch.zeros((B, T, H, H))
    true_jac[:, :] = W
    print(relative_error(true_jac, polf.jacs))
    print(" ----- ")

    print("Fundamental Matrices Relative Error:")
    true_U = torch.zeros((B, T+1, H, H))
    for t in range(T+1):
        true_U[:, t] = torch.linalg.matrix_power(W, t)
    print(relative_error(true_U, polf.Us)) 
    print(" ----- ")

    print("Trajectory Reconstruction Relative Error:")
    guess_z = polf(x) # Feeding x into the state-transition form should give perfect reconstruction since it's a linear ODE.
    print(relative_error(guess_z, hidden)) 
    print(" ----- ")

    from kpflow.frechet_op import FrechetOperator
    from kpflow.op_common import IdentityOperator as Id
    print("Check Pop Inverse if Frechet Operator:")
    frech = FrechetOperator(model_f, 0.*x, hidden)
    err1 = (frech @ polf).compare(Id(hidden.shape)) 
    err2 = (polf @ frech).compare(Id(hidden.shape)) 
    print(max(err1, err2)) 
    print(" ----- ")

def get_p_and_inv(model, inputs, hidden):
    cell = get_cell_from_model(model)
    pop = PropagationOperator_LinearForm(cell, inputs, hidden)
    pop_inv = FrechetOperator(cell, inputs, hidden)
    return pop, pop_inv


def test_operator_adjoints(plot = True, trials = 50):
    from kpflow.architecture import Model, get_cell_from_model
    from kpflow.parameter_op import ParameterOperator, JThetaOperator
    from kpflow.propagation_op import PropagationOperator_DirectForm, PropagationOperator_LinearForm
    from kpflow.op_common import AveragedOperator, check_adjoint

    # Test that the adjoint_call of the operators is indeed the adjoint, i.e. that 
    # <A x, y> = <x, A* y> for the parameter and propagation operators A.

    B, T, H = 5, 20, 10 # Batch size, Timesteps, Hidden count.
    model = Model(input_size = 3, output_size = 3, rnn=nn.GRU)

    inputs = torch.randn(B, T, 3)
    out, hidden = model(inputs)

    cell = get_cell_from_model(model)

    plt.figure(figsize = (12, 3))
    for idx, (name, struct) in enumerate(zip(['Direct P', 'Linearized P', 'K'], [PropagationOperator_DirectForm, PropagationOperator_LinearForm, ParameterOperator])):
        op = struct(cell, inputs, hidden)
        op_sp = op.to_scipy(hidden.shape, hidden.shape, dtype = float, can_matmat = False)

        plt.subplot(1,3,1+idx)
        t0 = perf_counter()
        errs = check_adjoint(op_sp, trials = trials)
        print(f'{name} compute time, {perf_counter() - t0}')

        if plot:
            plt.plot(errs)
            plt.title(f'{name} Operator')
            plt.xlabel('Random x, y Trial')
            if idx == 0:
                plt.ylabel('Relative Error')

    if plot:
        plt.suptitle('0 = |<A x, y> - <x, A* y>| Check')
        plt.tight_layout()

def test_projector_partial_trace_effdim():
    # Make the projector |Y><X| for random matrices, X, Y of shape (m, n)
    # Check tr_n(|Y><X|) = Y X^T (m, m) matrix and tr_m(|Y><X|) = Y^T X (n, n) matrix. 
    # Also check effdim_m(|Y><X|) = effrank(Y X^T) likewise for effdim_n.
    from kpflow.op_common import Projector, MatrixWrapper
    m, n = (10, 30)
    X,Y = torch.randn((m, n)), torch.rand((m, n))
    proj = Projector(X, Y) # proj(Q) = <Q, X>_F * Y projection

    V = X / np.linalg.norm(X, ord = 'fro')**2
    print(f'|Y><X|(X / ||X||) = Y, Frobenius relative error: {relative_error(Y, proj(V))}') 

    true_tr_n = MatrixWrapper(Y @ X.T)
    tr_n = proj.partial_trace(1).like(true_tr_n)
    print(f'Check tr_n(|Y><X|) = Y X^T, Frobenius relative error: {true_tr_n.compare(tr_n, nsamp = 50)}')

    true_tr_m = MatrixWrapper(Y.T @ X)
    tr_m = proj.partial_trace(0).like(true_tr_m)
    print(f'Check tr_m(|Y><X|) = Y^T X, Frobenius relative error: {true_tr_m.compare(tr_m, nsamp = 50)}')




if __name__ == '__main__':
    test_linearized_propagation()
    test_projector_partial_trace_effdim()
    test_operator_adjoints()
    plt.show()

