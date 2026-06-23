# Defining an implicit function, including its Jacobians and conversion methods.
import torch
import torch.nn as nn
from torch.func import functional_call
from .pytree_shape import ShapeSpec
from .op_common import GeneralOperator, JacobianOperator, WeightSiteOperator

# Implements an operator of the form 
# (state_space, param_space, optional input space)-> state space
# or re-ordered. User specifies state and param index in tuple.
class GlobalConstraint(GeneralOperator):
    # example_tuple_inp should be something like (h, theta, x)
    def __init__(self, call, example_tuple_inp, state_idx=0, param_idx=1, jacobians=None, **kwargs):
        self.state_idx = state_idx
        self.param_idx = param_idx
        self.example_tuple_inp = example_tuple_inp
        self.jacobians = {} if jacobians is None else dict(jacobians)

        shape_in = ShapeSpec.from_tree(example_tuple_inp)
        shape_out = ShapeSpec.from_tree(example_tuple_inp[state_idx])

        super().__init__(call, shape_in, shape_out, **kwargs)

    def _override(self, name, primals, default):
        op = self.jacobians.get(name)
        if op is None:
            return default()
        is_op = any(hasattr(op, attr) for attr in ("adjoint_call", "rmatvec", "solve", "rsolve"))
        return op(primals) if callable(op) and not is_op else op

    def _primals(self, primals):
        return self.example_tuple_inp if primals is None else primals

    def param_jac(self, primals=None):
        primals = self._primals(primals)
        return self._override(
            "param",
            primals,
            lambda: JacobianOperator(self, primals, argnums=self.param_idx, names=("theta", "F")),
        )

    def state_jac(self, primals=None):
        primals = self._primals(primals)
        return self._override(
            "state",
            primals,
            lambda: JacobianOperator(self, primals, argnums=self.state_idx, names=("h", "F")),
        )

    def jacs(self, primals=None):
        primals = self._primals(primals)
        return {
            "param": self.param_jac(primals),
            "state": self.state_jac(primals),
        }

    def propagator(self, primals=None, solver="neumann", **solver_kwargs):
        primals = self._primals(primals)
        return self.state_jac(primals).inverse(solver=solver, solver_kwargs = solver_kwargs)

    greens = propagator
    green = propagator

    # State-space neural tangent kernel.
    def ntk(self, primals=None, solver="neumann"):
        primals = self._primals(primals)
        theta_jac = self.param_jac(primals)
        prop = self.propagator(primals, solver=solver)
        return prop @ theta_jac @ theta_jac.T @ prop.T

    # State-space Fisher information.
    def fim(self, primals=None, solver="neumann"):
        primals = self._primals(primals)
        theta_jac = self.param_jac(primals)
        prop = self.propagator(primals, solver=solver)
        return theta_jac.T @ prop.T @ prop @ theta_jac


def _jacobian(call, primal, names):
    out = call(primal)
    op = GeneralOperator(
        lambda tpl: call(tpl[0]),
        ShapeSpec.from_tree((primal,)),
        ShapeSpec.from_tree(out),
    )
    return JacobianOperator(op, (primal,), argnums=0, names=names)


def _adjoint_call(op, x):
    if hasattr(op, "adjoint_call"):
        return op.adjoint_call(x)
    return op.rmatvec(x)


def _solve_adjoint(op, b, solver="neumann", **solver_kwargs):
    if hasattr(op, "rsolve"):
        return op.rsolve(b)
    solver_kwargs.setdefault("max_iter", b.shape[1] + 1 if b.ndim >= 2 else 1000)
    return op.T.solve(b, solver=solver, **solver_kwargs)


class WeightBasedOutputGradient:
    """Callable update from the block formula in WeightBasedOutputModel."""

    def __init__(self, ops, solver="neumann", **solver_kwargs):
        self.ops = ops
        self.solver = solver
        self.solver_kwargs = dict(solver_kwargs)

    def __call__(self, err):
        z = _solve_adjoint(self.ops["S"], _adjoint_call(self.ops["O"], err), self.solver, **self.solver_kwargs)
        update_W = _adjoint_call(self.ops["weight"], _adjoint_call(self.ops["B"], z))
        update_Wo = -_adjoint_call(self.ops["output_weight"], err)
        return update_W, update_Wo

    def __str__(self):
        return "WeightBasedOutputGradient"


class WeightBasedOutputModel:
    """
    Model in the form F((h,a,y),(W,W_o),x)=0:

        (G(h,a,x), a - weight_site(h,x) W.T, y - f_o(output_site(h) W_o.T))

    By default all Jacobians are inferred with JacobianOperator. Pass a
    jacobians dict to override any of:
        A, B, T, O, S, weight, output_weight
    Each override can be an operator or a callable taking primals.
    """

    def __init__(self, G, f_o, example, weight_site=None, output_site=None, jacobians=None):
        self.G = G
        self.f_o = f_o
        self.example = example
        self.weight_site = (lambda h, x: h) if weight_site is None else weight_site
        self.output_site = (lambda h: h) if output_site is None else output_site
        self.jacobians = {} if jacobians is None else dict(jacobians)

    def __call__(self, tpl):
        (h, a, y), (W, W_o), x = tpl
        return (
            self.G(h, a, x),
            a - self.weight_site(h, x) @ W.T,
            y - self.f_o(self.output_site(h) @ W_o.T),
        )

    def constraint(self):
        return GlobalConstraint(self, self.example, state_idx=0, param_idx=1)

    def _override(self, name, primals, default):
        op = self.jacobians.get(name)
        if op is None:
            return default()
        is_op = any(hasattr(op, attr) for attr in ("adjoint_call", "rmatvec", "solve", "rsolve"))
        return op(primals) if callable(op) and not is_op else op

    def ops(self, primals):
        (h, a, y), (W, W_o), x = primals
        site = self.weight_site(h, x)
        out_site = self.output_site(h)

        A = self._override("A", primals, lambda: _jacobian(lambda h_: self.G(h_, a, x), h, ("h", "G")))
        B = self._override("B", primals, lambda: _jacobian(lambda a_: self.G(h, a_, x), a, ("a", "G")))
        T = self._override("T", primals, lambda: _jacobian(lambda h_: self.weight_site(h_, x) @ W.T, h, ("h", "a")))
        O = self._override("O", primals, lambda: _jacobian(lambda h_: self.f_o(self.output_site(h_) @ W_o.T), h, ("h", "y")))
        S = self._override("S", primals, lambda: A + B @ T)
        weight = self._override("weight", primals, lambda: WeightSiteOperator(site, W.shape))
        output_weight = self._override("output_weight", primals, lambda: _jacobian(lambda Wo: self.f_o(out_site @ Wo.T), W_o, ("W_o", "y")))

        return {"A": A, "B": B, "T": T, "O": O, "S": S, "weight": weight, "output_weight": output_weight}

    def output_gradient(self, primals, solver="neumann", **solver_kwargs):
        return WeightBasedOutputGradient(self.ops(primals), solver=solver, **solver_kwargs)
