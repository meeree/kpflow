import torch
from torch import nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
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
    # Current equivalent of the old propagation operator test:
    # for a linear RNN h_t = W h_{t-1} + x_t, the Green's operator
    # (D_h F)^(-1) maps the input drive sequence to the hidden trajectory.
    from kpflow.architecture import BasicRNN

    B, T, H = 5, 20, 10
    x = torch.randn(B, T, H) * 1e-3
    W = torch.randn(H, H) / (H**0.5)
    rnn = BasicRNN(H, H, bias=False, linear=True)
    with torch.no_grad():
        rnn.cell.weight_hh.copy_(W)
        rnn.cell.weight_ih.copy_(torch.eye(H))

    hidden, _ = rnn(x)

    if plot:
        plt.plot(hidden[0, :, :])
        plt.show()

    F = rnn.to_implicit(x, h=hidden, jacobians="analytic")
    Dh = F.state_jac()
    P = F.greens(solver="neumann", max_iter=T + 1, tol=1e-10)

    print(" ----- ")
    print("Check D_hF inverse consistency:")
    q = torch.randn_like(hidden)
    inv_err = relative_error(Dh(P(q)), q)
    print(inv_err)
    assert inv_err < 1e-5
    print(" ----- ")

def get_p_and_inv(model, inputs, hidden):
    F = model.to_implicit(inputs, h=hidden)
    pop_inv = F.state_jac()
    pop = F.greens(solver="neumann", max_iter=inputs.shape[1] + 1, tol=1e-10)
    return pop, pop_inv


def _tree_inner(a, b):
    return sum((a[k] * b[k]).sum() for k in a)


def test_operator_adjoints(plot = False, trials = 10):
    from kpflow.architecture import BasicRNN

    B, T, N_in, H = 4, 7, 3, 6
    model = BasicRNN(N_in, H, bias=True)
    inputs = torch.randn(B, T, N_in)
    hidden, _ = model(inputs)
    F = model.to_implicit(inputs, h=hidden, jacobians="analytic")

    for name, op in [("state", F.state_jac()), ("parameter", F.param_jac())]:
        errs = []
        for _ in range(trials):
            y = torch.randn_like(hidden)
            if name == "state":
                x = torch.randn_like(hidden)
                lhs = (op(x) * y).sum()
                rhs = (x * op.adjoint_call(y)).sum()
            else:
                theta = dict(model.cell.named_parameters())
                x = {k: torch.randn_like(v) for k, v in theta.items()}
                lhs = (op(x) * y).sum()
                rhs = _tree_inner(x, op.adjoint_call(y))
            errs.append((lhs - rhs).abs() / (lhs.abs() + rhs.abs() + 1.0))
        err = torch.stack(errs).max().item()
        print(f"{name} operator adjoint relative error:", err)
        assert err < 1e-5

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




def _max_param_tree_error(a, b):
    return max((a[k] - b[k]).abs().max().item() for k in a)


def test_basic_rnn_matches_manual_rollout():
    from kpflow.architecture import BasicRNN

    torch.manual_seed(0)
    B, T, N_in, N = 3, 5, 4, 6
    model = BasicRNN(N_in, N, bias=True)
    x = torch.randn(B, T, N_in)

    hidden, _ = model(x)
    h = torch.zeros(B, N)
    manual = []
    for t in range(T):
        h = torch.tanh(h) @ model.cell.weight_hh.T + x[:, t] @ model.cell.weight_ih.T + model.cell.bias
        manual.append(h)
    manual = torch.stack(manual, 1)

    err = (hidden - manual).abs().max().item()
    print("BasicRNN manual rollout max error:", err)
    assert err < 1e-6


def test_basic_rnn_to_implicit_dream_syntax():
    from kpflow.architecture import Model, BasicRNN

    torch.manual_seed(1)
    B, T, N_in, N, N_out = 2, 4, 3, 5, 2
    model = Model(N_in, N, N_out, rnn=BasicRNN, bias=True)
    x = torch.randn(B, T, N_in)
    _, h = model(x)

    implicit_model = model.to_implicit(x)
    residual_err = implicit_model(implicit_model.example_tuple_inp).abs().max().item()
    Dh = implicit_model.state_jac()
    Dtheta = implicit_model.param_jac()
    P = implicit_model.greens(solver="neumann", max_iter=T + 1, tol=1e-10)

    w = torch.randn_like(h)
    inverse_err = (Dh(P(w)) - w).abs().max().item()
    print("to_implicit residual max error:", residual_err)
    print("to_implicit inverse max error:", inverse_err)
    print("to_implicit state shape:", Dh.shape_in, "->", Dh.shape_out)
    print("to_implicit param shape:", Dtheta.shape_in, "->", Dtheta.shape_out)

    assert residual_err < 1e-6
    assert inverse_err < 1e-5


def test_basic_rnn_analytic_jacobians_match_default_implicit():
    from kpflow.architecture import BasicRNN

    torch.manual_seed(2)
    B, T, N_in, N = 2, 4, 3, 5
    rnn = BasicRNN(N_in, N, bias=True)
    x = torch.randn(B, T, N_in)
    h, _ = rnn(x)

    F_default = rnn.to_implicit(x, h=h)
    F_analytic = rnn.to_implicit(x, h=h, jacobians="analytic")

    Dh_default = F_default.state_jac()
    Dh_analytic = F_analytic.state_jac()
    Dtheta_default = F_default.param_jac()
    Dtheta_analytic = F_analytic.param_jac()

    dh = torch.randn_like(h)
    w = torch.randn_like(h)
    theta = dict(rnn.cell.named_parameters())
    dtheta = {k: torch.randn_like(v) for k, v in theta.items()}

    state_mv_err = (Dh_default(dh) - Dh_analytic(dh)).abs().max().item()
    state_adj_err = (Dh_default.adjoint_call(w) - Dh_analytic.adjoint_call(w)).abs().max().item()
    param_mv_err = (Dtheta_default(dtheta) - Dtheta_analytic(dtheta)).abs().max().item()
    param_adj_err = _max_param_tree_error(Dtheta_default.adjoint_call(w), Dtheta_analytic.adjoint_call(w))

    print("BasicRNN analytic state matvec max error:", state_mv_err)
    print("BasicRNN analytic state adjoint max error:", state_adj_err)
    print("BasicRNN analytic param matvec max error:", param_mv_err)
    print("BasicRNN analytic param adjoint max error:", param_adj_err)

    assert state_mv_err < 1e-6
    assert state_adj_err < 1e-6
    assert param_mv_err < 1e-6
    assert param_adj_err < 1e-6


if __name__ == '__main__':
    test_basic_rnn_matches_manual_rollout()
    test_basic_rnn_to_implicit_dream_syntax()
    test_basic_rnn_analytic_jacobians_match_default_implicit()
    test_linearized_propagation()
    test_projector_partial_trace_effdim()
    test_operator_adjoints()
    plt.show()
