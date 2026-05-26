import torch
import torch.nn as nn
from torch.func import functional_call

from kpflow.implicit import GlobalConstraint


GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def print_test(name, passed, extra=""):
    status = "PASSED" if passed else "FAILED"
    color = GREEN if passed else RED

    print(f"******* TEST {name}: {color}{status}{RESET} *****")
    if extra:
        print(extra)


def max_abs_tree(tree_a, tree_b):
    vals = []
    for k in tree_a:
        vals.append((tree_a[k] - tree_b[k]).abs().max())
    return torch.stack(vals).max().item()


def rel_norm_tree(tree_a, tree_b, eps=1e-12):
    num = []
    den = []
    for k in tree_a:
        num.append((tree_a[k] - tree_b[k]).norm())
        den.append(tree_a[k].norm())
    num = torch.stack(num).norm()
    den = torch.stack(den).norm()
    return (num / (den + eps)).item()


def step_down(h):
    """
    h: [B, T, N]
    returns shifted h_{t-1}, with zero at t=0.
    """
    return torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)


class TinyRNNCell(nn.Module):
    def __init__(self, n_in, n_hidden):
        super().__init__()
        self.W_h = nn.Linear(n_hidden, n_hidden, bias=False)
        self.W_x = nn.Linear(n_in, n_hidden, bias=False)

    def forward(self, h_prev, x):
        return torch.tanh(self.W_h(h_prev) + self.W_x(x))


def explicit_rnn_rollout(model, theta, x):
    """
    Explicit sequential rollout.

    x: [B, T, N_in]
    h: [B, T, N]
    """
    B, T, _ = x.shape
    N = model.W_h.out_features

    h_prev = torch.zeros(B, N, dtype=x.dtype, device=x.device)
    hs = []

    for t in range(T):
        h_prev = functional_call(model, theta, (h_prev, x[:, t]))
        hs.append(h_prev)

    return torch.stack(hs, dim=1)


def implicit_F(model):
    """
    Returns single-pytree-argument version of:

        F(h, theta, x) = h - model(step_down(h), x, theta)

    h:     [B, T, N]
    theta: dict of module params
    x:     [B, T, N_in]
    """

    def F(h, theta, x):
        h_prev = step_down(h)
        pred = functional_call(model, theta, (h_prev, x))
        return h - pred

    return lambda tpl: F(*tpl)


def test_rnn_global_constraint():
    torch.manual_seed(0)

    B = 4
    T = 6
    N_in = 3
    N = 5

    tol_residual = 1e-6
    tol_inverse = 1e-5
    tol_update_abs = 1e-5
    tol_update_rel = 1e-5

    model = TinyRNNCell(N_in, N)

    # Make recurrent dynamics mild.
    with torch.no_grad():
        model.W_h.weight.mul_(0.3)
        model.W_x.weight.mul_(0.5)

    theta = dict(model.named_parameters())
    x = torch.randn(B, T, N_in)

    # Explicit rollout.
    h = explicit_rnn_rollout(model, theta, x)

    # Define implicit constraint.
    F = GlobalConstraint(
        implicit_F(model),
        (h, theta, x),
        state_idx=0,
        param_idx=1,
    )

    primals = (h, theta, x)

    # ------------------------------------------------------------
    # Test 1: explicit rollout satisfies implicit residual.
    # ------------------------------------------------------------
    residual = F(primals)
    residual_err = residual.abs().max().item()
    passed = residual_err < tol_residual

    print()
    print("residual shape:", residual.shape)
    print("max |F(h, theta, x)|:", residual_err)
    print_test("IMPLICIT RESIDUAL", passed)

    assert passed

    # ------------------------------------------------------------
    # Build Jacobians.
    # ------------------------------------------------------------
    DhF = F.state_jac(primals)
    DthetaF = F.param_jac(primals)

    print()
    print("DhF:", DhF)
    print("DhF.shape_in: ", DhF.shape_in)
    print("DhF.shape_out:", DhF.shape_out)

    print()
    print("DthetaF:", DthetaF)
    print("DthetaF.shape_in: ", DthetaF.shape_in)
    print("DthetaF.shape_out:", DthetaF.shape_out)

    # ------------------------------------------------------------
    # Test 2: state Jacobian matvec shape.
    # ------------------------------------------------------------
    dh = torch.randn_like(h)
    Jdh = DhF(dh)

    passed = Jdh.shape == h.shape
    print()
    print("DhF(dh) shape:", Jdh.shape)
    print_test("STATE JACOBIAN MATVEC SHAPE", passed)

    assert passed

    # ------------------------------------------------------------
    # Test 3: state Jacobian adjoint shape.
    # ------------------------------------------------------------
    w = torch.randn_like(h)
    JT_w = DhF.adjoint_call(w)

    passed = JT_w.shape == h.shape
    print()
    print("DhF.T(w) shape:", JT_w.shape)
    print_test("STATE JACOBIAN ADJOINT SHAPE", passed)

    assert passed

    # ------------------------------------------------------------
    # Test 4: parameter Jacobian matvec shape.
    # ------------------------------------------------------------
    dtheta = {
        name: torch.randn_like(p)
        for name, p in theta.items()
    }

    Jdtheta = DthetaF(dtheta)

    passed = Jdtheta.shape == h.shape
    print()
    print("DthetaF(dtheta) shape:", Jdtheta.shape)
    print_test("PARAM JACOBIAN MATVEC SHAPE", passed)

    assert passed

    # ------------------------------------------------------------
    # Test 5: parameter Jacobian adjoint shape.
    # ------------------------------------------------------------
    grad_theta = DthetaF.adjoint_call(w)

    passed = True
    for name, p in theta.items():
        if name not in grad_theta:
            passed = False
        elif grad_theta[name].shape != p.shape:
            passed = False

    print()
    print("DthetaF.T(w) shapes:")
    for k, v in grad_theta.items():
        print(f"  {k}: {tuple(v.shape)}")
    print_test("PARAM JACOBIAN ADJOINT SHAPE", passed)

    assert passed

    # ------------------------------------------------------------
    # Build Green's operator P = (D_h F)^(-1).
    # ------------------------------------------------------------
    P = F.greens(
        primals,
        solver="neumann",
        max_iter=T + 1,
        tol=1e-10,
    )

    print()
    print("P = (DhF)^(-1)")
    print("P.shape_in: ", P.shape_in)
    print("P.shape_out:", P.shape_out)

    # ------------------------------------------------------------
    # Test 6: inverse consistency DhF(Pw) = w.
    # ------------------------------------------------------------
    Pw = P(w)
    inverse_err = (DhF(Pw) - w).abs().max().item()
    passed = inverse_err < tol_inverse

    print()
    print("max |DhF(Pw) - w|:", inverse_err)
    print_test("GREEN INVERSE CONSISTENCY", passed)

    assert passed

    # ------------------------------------------------------------
    # Test 7: explicit BPTT update matches implicit adjoint update.
    #
    # F(h, theta, x) = 0
    #
    # D_h F dh + D_theta F dtheta = 0
    # dh/dtheta = - (D_h F)^(-1) D_theta F
    #
    # grad_theta L
    #   = (dh/dtheta)^* grad_h L
    #   = - D_theta F^* (D_h F)^(-*) grad_h L
    #
    # With LR = 1:
    #
    #   update = - grad_theta L
    #          = D_theta F^* (D_h F)^(-*) grad_h L
    # ------------------------------------------------------------
    target = torch.randn_like(h)

    for p in model.parameters():
        p.grad = None

    h_bptt = explicit_rnn_rollout(model, theta, x)
    loss = 0.5 * ((h_bptt - target) ** 2).sum()
    loss.backward()

    explicit_update = {
        name: -p.grad.detach().clone()
        for name, p in model.named_parameters()
    }

    # err_h = grad_h L for L = 0.5 ||h - target||^2
    err_h = h.detach() - target.detach()

    # lambda = P^* err_h = (D_h F)^(-*) err_h
    lam = P.adjoint_call(err_h)

    # update = D_theta F^* lambda
    implicit_update = DthetaF.adjoint_call(lam)

    update_abs_err = max_abs_tree(explicit_update, implicit_update)
    update_rel_err = rel_norm_tree(explicit_update, implicit_update)

    passed = (
        update_abs_err < tol_update_abs
        and update_rel_err < tol_update_rel
    )

    print()
    print("Compare explicit BPTT update vs implicit adjoint update:")
    for name in explicit_update:
        a = explicit_update[name]
        b = implicit_update[name]

        abs_err = (a - b).abs().max().item()
        rel_err = (a - b).norm().item() / (a.norm().item() + 1e-12)

        print(f"  {name}:")
        print(f"    explicit update shape: {tuple(a.shape)}")
        print(f"    implicit update shape: {tuple(b.shape)}")
        print(f"    max abs err: {abs_err:.3e}")
        print(f"    rel err:     {rel_err:.3e}")

    print("max update abs err:", update_abs_err)
    print("max update rel err:", update_rel_err)
    print_test("BPTT UPDATE MATCHES IMPLICIT ADJOINT UPDATE", passed)

    assert passed

    return F, DhF, DthetaF, P


if __name__ == "__main__":
    F, DhF, DthetaF, P = test_rnn_global_constraint()
