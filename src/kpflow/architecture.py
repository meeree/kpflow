import torch
from torch import nn
from torch.func import functional_call

from .implicit import GlobalConstraint
from .op_common import LinearOperator
from .pytree_shape import ShapeSpec


def step_down(h):
    return torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)


def _activation_pair(activation):
    if activation in (None, "identity", "linear"):
        return (lambda x: x), (lambda x: torch.ones_like(x))
    if activation in (torch.tanh, "tanh"):
        return torch.tanh, lambda x: 1.0 - torch.tanh(x) ** 2
    if activation in (torch.relu, "relu"):
        return torch.relu, lambda x: (x > 0).to(x.dtype)
    if activation in (torch.sigmoid, "sigmoid"):
        return torch.sigmoid, lambda x: torch.sigmoid(x) * (1.0 - torch.sigmoid(x))
    return activation, None


def _cell_step(cell, h, x):
    if isinstance(cell, RecurrentCell):
        return cell(h, x)
    return cell(x, h)


class RecurrentCell(nn.Module):
    """Cell interface for h_{t+1} = f(h_t, x_{t+1}, theta)."""

    def initial_state(self, x):
        return torch.zeros((x.shape[0], self.hidden_size), dtype=x.dtype, device=x.device)

    def rollout(self, x, h0=None):
        h = self.initial_state(x) if h0 is None else h0
        if h.ndim == 3 and h.shape[0] == 1:
            h = h[0]
        hidden = []
        for t in range(x.shape[1]):
            h = self(h, x[:, t])
            hidden.append(h)
        return torch.stack(hidden, dim=1)

    def residual(self, h, theta, x):
        return h - functional_call(self, theta, (step_down(h), x))

    def analytic_jacobians(self, primals):
        return None

    def to_implicit(self, x, h=None, theta=None, jacobians=None):
        theta = dict(self.named_parameters()) if theta is None else theta
        h = self.rollout(x) if h is None else h
        primals = (h, theta, x)
        if jacobians == "analytic":
            jacobians = self.analytic_jacobians(primals)
        return GlobalConstraint(lambda tpl: self.residual(*tpl), primals, state_idx=0, param_idx=1, jacobians=jacobians)


class _RNNStateJacobian(LinearOperator):
    def __init__(self, h, W, dsigma):
        self.W = W
        self.dsigma_h = dsigma(h)
        super().__init__(h.shape, h.shape, dev=h.device)

    def _matvec(self, dh):
        return dh - step_down(self.dsigma_h * dh) @ self.W.T

    def _rmatvec(self, w):
        future = torch.cat([w[:, 1:], torch.zeros_like(w[:, :1])], dim=1)
        return w - (future @ self.W) * self.dsigma_h

    def __str__(self):
        return "D_hF[BasicRNN]"


class _RNNParamJacobian(LinearOperator):
    def __init__(self, h, theta, x, sigma):
        self.site_h = sigma(step_down(h))
        self.theta = theta
        self.x = x
        super().__init__(ShapeSpec.from_tree(theta), h.shape, dev=h.device)

    def _matvec(self, dtheta):
        out = torch.zeros((*self.x.shape[:-1], self.shape_out[-1]), dtype=self.x.dtype, device=self.x.device)
        out = out - self.site_h @ dtheta["weight_hh"].T
        out = out - self.x @ dtheta["weight_ih"].T
        if "bias" in dtheta:
            out = out - dtheta["bias"]
        return out

    def _rmatvec(self, w):
        grad = {key: torch.zeros_like(param) for key, param in self.theta.items()}
        grad["weight_hh"] = -torch.einsum("btn,btm->nm", w, self.site_h)
        grad["weight_ih"] = -torch.einsum("btn,bti->ni", w, self.x)
        if "bias" in self.theta:
            grad["bias"] = -w.sum(dim=(0, 1))
        return grad

    def __str__(self):
        return "D_thetaF[BasicRNN]"


class BasicRNNCell(RecurrentCell):
    def __init__(self, n_in, n, bias=True, activation=torch.tanh, linear=False, **kwargs):
        super().__init__()
        if kwargs:
            raise TypeError(f"Unsupported BasicRNNCell kwargs: {sorted(kwargs)}")
        if linear:
            activation = "identity"
        self.input_size = n_in
        self.hidden_size = n
        self.weight_ih = nn.Parameter(torch.empty(n, n_in))
        self.weight_hh = nn.Parameter(torch.empty(n, n))
        self.bias = nn.Parameter(torch.empty(n)) if bias else None
        self.sigma, self.dsigma = _activation_pair(activation)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_ih, a=5 ** 0.5)
        nn.init.orthogonal_(self.weight_hh)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, h, x=None):
        if x is None:
            x, h = h, None
        elif h.shape[-1] == self.input_size and x.shape[-1] == self.hidden_size:
            x, h = h, x
        if h is None:
            h = torch.zeros((*x.shape[:-1], self.hidden_size), dtype=x.dtype, device=x.device)
        out = self.sigma(h) @ self.weight_hh.T + x @ self.weight_ih.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def analytic_jacobians(self, primals):
        if self.dsigma is None:
            return None
        h, theta, x = primals
        return {
            "state": lambda p: _RNNStateJacobian(p[0], p[1]["weight_hh"], self.dsigma),
            "param": lambda p: _RNNParamJacobian(p[0], p[1], p[2], self.sigma),
        }


class BasicRNN(nn.Module):
    def __init__(self, n_in, n, batch_first=True, bias=True, cell=None, **cell_kwargs):
        super().__init__()
        self.cell = BasicRNNCell(n_in, n, bias=bias, **cell_kwargs) if cell is None else cell
        self.batch_first = batch_first
        self.input_size = self.cell.input_size
        self.hidden_size = self.cell.hidden_size
        self.bias = self.cell.bias is not None
        self.weight_ih_l0 = self.cell.weight_ih
        self.weight_hh_l0 = self.cell.weight_hh
        if self.bias:
            self.bias_l0 = self.cell.bias

    def forward(self, x, h0=None):
        x_in = x if self.batch_first else x.swapaxes(0, 1)
        hidden = self.cell.rollout(x_in, h0)
        if not self.batch_first:
            hidden = hidden.swapaxes(0, 1)
        return hidden, None

    def to_implicit(self, x, h=None, theta=None, jacobians=None):
        return self.cell.to_implicit(x, h=h, theta=theta, jacobians=jacobians)


class Model(nn.Module):
    def __init__(self, input_size=3, hidden_size=100, output_size=3, rnn=nn.GRU, bias=True, **kwargs):
        super().__init__()
        if rnn in (nn.RNN, nn.RNNCell):
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, bias=bias, **kwargs)
        elif rnn in (nn.GRU, nn.GRUCell):
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, bias=bias, **kwargs)
        elif rnn in (BasicRNN, BasicRNNCell, RecurrentCell):
            self.rnn = BasicRNN(input_size, hidden_size, bias=bias, **kwargs)
        elif isinstance(rnn, type) and issubclass(rnn, RecurrentCell):
            try:
                cell = rnn(input_size, hidden_size, bias=bias, **kwargs)
            except TypeError:
                cell = rnn(input_size, hidden_size, **kwargs)
            self.rnn = BasicRNN(input_size, hidden_size, cell=cell)
        else:
            self.rnn = rnn(input_size, hidden_size, batch_first=True, bias=bias, **kwargs)

        self.Wout = nn.Linear(hidden_size, output_size, bias=bias)
        self.hidden_size = hidden_size

    def forward(self, x, h0=None):
        hidden, _ = self.rnn(x, h0)
        return self.Wout(hidden), hidden

    def to_implicit(self, x, h=None, theta=None, jacobians=None):
        if not hasattr(self.rnn, "to_implicit"):
            raise TypeError("to_implicit is only available for RecurrentCell-based models such as BasicRNN.")
        if h is None:
            _, h = self(x)
        return self.rnn.to_implicit(x, h=h, theta=theta, jacobians=jacobians)

    def analysis_mode(self, X, target, h0=None, return_param_grads=False):
        cell = get_cell_from_model(self)
        for param in cell.parameters():
            if param.grad is not None:
                param.grad.zero_()
            param.requires_grad_()

        hidden = [h0 if h0 is not None else torch.zeros((X.shape[0], self.hidden_size), device=X.device)]
        for t in range(X.shape[1]):
            hidden.append(_cell_step(cell, hidden[-1], X[:, t]).clone())
        hidden = hidden[1:]
        for h in hidden:
            h.requires_grad_()
            h.retain_grad()
        hidd_stack = torch.stack(hidden, 1)
        out = self.Wout(hidd_stack)
        loss_unreduced = nn.MSELoss(reduction="none")(out, target)
        err = torch.autograd.grad(torch.sum(loss_unreduced), hidd_stack, retain_graph=True)[0]
        loss = loss_unreduced.sum()
        loss.backward()
        adjoint = torch.stack([h.grad for h in hidden], 1)
        hidden = torch.stack(hidden, 1)

        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()
            param.requires_grad_()

        out = self(X)[0]
        loss = nn.MSELoss(reduction="sum")(out, target)
        loss.backward()
        if return_param_grads:
            return hidden, adjoint, err, out, loss_unreduced, loss, {
                name: param.grad.clone()
                for name, param in self.named_parameters()
            }
        return hidden, adjoint, err, out, loss_unreduced, loss


def get_cell_from_model(model):
    if isinstance(model.rnn, nn.GRU):
        cell = nn.GRUCell(model.rnn.input_size, model.rnn.hidden_size, bias=model.rnn.bias).to(model.Wout.weight.device)
    elif isinstance(model.rnn, BasicRNN):
        return model.rnn.cell
    elif isinstance(model.rnn, RecurrentCell):
        return model.rnn
    else:
        cell = nn.RNNCell(model.rnn.input_size, model.rnn.hidden_size, bias=model.rnn.bias).to(model.Wout.weight.device)

    cell.weight_ih.data.copy_(model.rnn.weight_ih_l0.data)
    cell.weight_hh.data.copy_(model.rnn.weight_hh_l0.data)
    if model.rnn.bias:
        cell.bias_ih.data.copy_(model.rnn.bias_ih_l0.data)
        cell.bias_hh.data.copy_(model.rnn.bias_hh_l0.data)
    return cell
