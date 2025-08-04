# Utilities for computing quantities related to the Lyapunov exponents. 
import torch
import numpy as np

# Utility function
np_to_torch = lambda x: x if torch.is_tensor(x) else torch.from_numpy(x)
torch_to_np = lambda x: x if not torch.is_tensor(x) else x.detach().cpu().numpy()

@torch.no_grad
def compute_jacobians(model_f, hidden, to_np = False):
    # Hidden is shape [batches, time, hidden].
    # Suppose we want partials df(h(t,b)) / dh(t,b) for all t, b. 
    # Note that d(sum_t sum_b f(h(t,b))) / dh(t,b) = df(h(t,b,)) / dh(t,b), i.e. we can sum over all time and batches and get same partials.

    # model_f should just accept a single input z. If it accepts multiple inputs, use functools.partial or a lambda!
    H = hidden.shape[-1]
    hidden_flat = hidden.reshape((-1, H))

    model_f_sum = lambda z: model_f(z).sum(0)
    jacs = torch.autograd.functional.jacobian(model_f_sum, hidden_flat.detach().clone())
    jacs = torch.movedim(jacs, 0, -2).reshape((*hidden.shape[:-1], H, H)) # Shape (B, T, H, H). Multiply on the left.  
    return torch_to_np(jacs) if to_np else jacs

@torch.no_grad
def fundamental_matrix(jacs, return_U = True, debug = False):
    # jacs is assumed to be shape [..., time, n, n] where n is state dimension.
    # Iteratively compute:
    # M(t+1) = J(t) * Q(t), [Q(t+1), R(t+1)] = qr(M(t+1)).
    # Fundamental matrix is given by U(t) = Q(t) * R(t) * R(t-1) ... * R(1).
    n = jacs.shape[-1]
    eye_like = torch.zeros_like(jacs[..., 0, :, :])
    eye_like[..., range(n), range(n)] = 1 # Shape [..., n, n] where each nxn matrix is diagonal.

    Rs, Qs, Rs_cum = [], [], [] # All the matrices to record over time.
    Q = eye_like.clone()
    for J in jacs.moveaxis(-3, 0): # Iterate over time.
        M = J @ Q
        Q, R = torch.linalg.qr(M)
        R = R + 1e-10 * eye_like # Add a "nugget"
        Rs.append(R.clone())
        Qs.append(Q.clone())

    Rs = torch.stack(Rs, 0) # [time, ...,  n, n].
#    scales = torch.linalg.norm(Rs, dim = (-2, -1), keepdim = True) # ||R_i|| for each R_i.
#    Rs_nm = Rs / scales # R_i / ||R_i||, i.e. normalized.
#    scales_cum = torch.exp(torch.cumsum(torch.log(scales), 0)) # S_i = exp(Sum_j^i log(||R_j||) = prod^i(||R_j||), but more stable since we add instead of multiply.
#
#    R_cum_nm = eye_like.clone()
#    for Q, R_nm, scale in zip(Qs, Rs_nm, scales_cum):
#        R_cum_nm = R_nm @ R_cum_nm
#        Rs_cum.append(scale * R_cum_nm)

    if debug:
        extrema = torch.abs(Rs)[:, :, range(n), range(n)].min(0)[0]
        print(extrema.shape)
        import matplotlib.pyplot as plt
        plt.imshow(extrema, aspect='auto')
        plt.colorbar()
        plt.show()

        plt.plot(extrema[:, -1])
        plt.show()

    R_cum = eye_like.clone()
    for Q, R in zip(Qs, Rs):
        R_cum = R @ R_cum
        Rs_cum.append(R_cum.clone())
    Rs_cum = torch.stack(Rs_cum, -3)

    Qs = torch.stack(Qs, -3) # [..., time, n, n].
    if not return_U:
        return Qs, Rs_cum

    Us = Qs @ Rs_cum 

    Us = [eye_like.clone()]
    for i in range(jacs.shape[-3]):
        Us.append(jacs[..., i, :, :] @ Us[-1])
    Us = torch.stack(Us, -3)

    return Qs, Rs, Us
    
@torch.no_grad
def state_transition(Q, R, i1, i2):
    # U [i1] = Q[i1] * R_cum[i1], U[i2] = Q[i2] * R_cum[i2], so 
    # Phi(i1, i2) =Q[i1] * R_cum[i1] * (R_cum[i2]^{-1}) * Q[i2].T.
    prod = Q[:, i1]
    for i in range(i1, i2, -1):
        prod = prod @ R[i]
    return prod @ Q[:, i2].swapaxes(-2, -1)
