# Defining an implicit function, including its Jacobians and conversion methods.
import torch
import torch.nn as nn
from torch.func import functional_call
from .pytree_shape import ShapeSpec
from .op_common import GeneralOperator, JacobianOperator

# Implements an operator of the form 
# (state_space, param_space, optional input space)-> state space
# or re-ordered. User specifies state and param index in tuple.
class GlobalConstraint(GeneralOperator):
    # example_tuple_inp should be something like (h, theta, x)
    def __init__(self, call, example_tuple_inp, state_idx=0, param_idx=1, **kwargs):
        self.state_idx = state_idx
        self.param_idx = param_idx

        shape_in = ShapeSpec.from_tree(example_tuple_inp)
        shape_out = ShapeSpec.from_tree(example_tuple_inp[state_idx])

        super().__init__(call, shape_in, shape_out, **kwargs)

    def param_jac(self, primals):
        return JacobianOperator(self, primals, argnums=self.param_idx, names=("theta", "F"))

    def state_jac(self, primals):
        return JacobianOperator(self, primals, argnums=self.state_idx, names=("h", "F"))

    def jacs(self, primals):
        return {
            "param": self.param_jac(primals),
            "state": self.state_jac(primals),
        }

    def propagator(self, primals, solver="neumann", **solver_kwargs):
        return self.state_jac(primals).inverse(solver=solver, solver_kwargs = solver_kwargs)

    greens = propagator
