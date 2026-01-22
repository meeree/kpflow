from torch import nn
import torch

class Model(nn.Module):
    def __init__(self, input_size=3, hidden_size=100, output_size=3, rnn = nn.GRU, bias = True, **kwargs):
        super().__init__()
        if rnn == nn.RNN or rnn == nn.RNNCell:
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, bias = bias, **kwargs)
        elif rnn == nn.GRU or rnn == nn.GRUCell:
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, bias = bias, **kwargs)
        else:
            self.rnn = BasicRNN(input_size, hidden_size, **kwargs)

        self.Wout = nn.Linear(hidden_size, output_size, bias = bias)
        self.hidden_size = hidden_size
        
    def forward(self, x, h0=None):
        hidden, _ = self.rnn(x, h0)
        return self.Wout(hidden), hidden  # [B, T, 3]
    
    def analysis_mode(self, X, target, h0=None, return_param_grads = False):
        # Intended for deep analysis of the GD flow:
        # Run RNN and compute losses, 
        # returning hidden, adjoints, outputs, unreduced losses, reduced loss.
        # Shapes: [B, T, H], [B, T, H], [B, T, O], [B, T, O], scalar.
        cell = get_cell_from_model(self)
        for param in cell.parameters():
            if param.grad is not None:
                param.grad.zero_()
            param.requires_grad_()
        
        hidden = [h0 if h0 is not None else torch.zeros((X.shape[0], self.hidden_size))]
        hidden[0] = hidden[0].to(X.device)
        for t in range(X.shape[1]):
            hidden.append(cell(X[:, t], hidden[-1]).clone())
        hidden = hidden[1:]
        for h in hidden:
            h.requires_grad_()
            h.retain_grad()
        hidd_stack = torch.stack(hidden, 1)
        out = self.Wout(hidd_stack) # [B, T, output_size]
        loss_fn = nn.MSELoss(reduction='none')
        loss_unreduced = loss_fn(out, target)
        err = torch.autograd.grad(torch.sum(loss_unreduced), hidd_stack, retain_graph = True)[0]
        loss = loss_unreduced.sum()
        loss.backward()  # Perform BPTT.
        adjoint = torch.stack([h.grad for h in hidden], 1)  # dL/dz defn of adjoint.
        hidden = torch.stack(hidden, 1)

        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()
            param.requires_grad_()

        out = self(X)[0]
        loss_fn = nn.MSELoss(reduction='sum')
        loss = loss_fn(out, target)
        loss.backward()
        if return_param_grads:
            param_grads = {}
            for name, param in self.named_parameters():
                param_grads[name] = param.grad.clone()
            return hidden, adjoint, err, out, loss_unreduced, loss, param_grads

#        if return_param_grads:
#            param_grads = {}
#            for name, param in cell.named_parameters():
#                param_grads[name] = param.grad.clone()
#            
#            return hidden, adjoint, err, out, loss_unreduced, loss, param_grads
        
        return hidden, adjoint, err, out, loss_unreduced, loss

class BasicRNNCell(nn.Module):
    def __init__(self, n_in, n, linear = False, noise_std = None, dt = 1.):
        super().__init__()
        self.bias = False
        self.weight_ih = nn.Linear(n_in, n, bias = False).weight
        self.weight_hh = nn.Linear(n, n, bias = False).weight
        self.input_size = n_in
        self.hidden_size = n
        self.actvn = (lambda x: x) if linear else torch.tanh
        self.noise = (lambda _: 0.) if noise_std is None else (lambda shape: torch.randn(shape).to(self.weight_ih.device) * noise_std)
        self.dt = dt

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros((x.shape[0], self.hidden_size)).to(x.device)
        return (1-self.dt) * h + self.dt * (self.actvn(h) @ self.weight_hh.T + x @ self.weight_ih.T + self.noise(h.shape))

class BasicRNN(nn.Module):
    def __init__(self, n_in, n, batch_first = True, linear = False, noise_std = None, dt = 1.):
        super().__init__()
        self.bias = False
        self.cell = BasicRNNCell(n_in, n, linear = linear, noise_std = noise_std, dt = dt)
        self.batch_first = batch_first
        self.hidden_size = n
        self.input_size = n_in
        self.weight_ih_l0 = self.cell.weight_ih
        self.weight_hh_l0 = self.cell.weight_hh

    def forward(self, x, h=None):
        x_itr = x
        if self.batch_first:
            x_itr = x.swapaxes(0,1) # Put time first.

        hidden = [self.cell(x_itr[0], h)]
        for x_t in x_itr[1:]:
            hidden.append(self.cell(x_t, hidden[-1].clone()))
        return torch.stack(hidden, (1 if self.batch_first else 0)), None

# Get a GRUCell from the model above with the same parameters.
def get_cell_from_model(model):
    if isinstance(model.rnn, nn.GRU):
        cell = nn.GRUCell(model.rnn.input_size, model.rnn.hidden_size, bias = model.rnn.bias).to(model.Wout.weight.device)
    elif isinstance(model.rnn, BasicRNN):
        return model.rnn.cell
    else:
        cell = nn.RNNCell(model.rnn.input_size, model.rnn.hidden_size, bias = model.rnn.bias).to(model.Wout.weight.device)

    cell.weight_ih.data.copy_(model.rnn.weight_ih_l0.data)
    cell.weight_hh.data.copy_(model.rnn.weight_hh_l0.data)
    if model.rnn.bias:  
        cell.bias_ih.data.copy_(model.rnn.bias_ih_l0.data)
        cell.bias_hh.data.copy_(model.rnn.bias_hh_l0.data)
    return cell
