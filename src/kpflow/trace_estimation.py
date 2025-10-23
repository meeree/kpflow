import numpy as np
import matplotlib.pyplot as plt
import torch

# Hutch++ Algorithm.
@torch.no_grad()
def trace_hupp(A, nsamp):
    d = A.shape[0]
    S = np.random.randint(2, size=(d,nsamp)).astype(float) * 2 - 1 # Either 1 or -1
    G = np.random.randint(2, size=(d,nsamp)).astype(float) * 2 - 1 # Either 1 or -1

    Q, _ = np.linalg.qr(A @ S)
    prod = G - Q @ (Q.T @ G)
    return np.trace((A @ Q).T @ Q) + (1./nsamp) * np.trace((A @ prod).T @ prod)

# Version using my Operator class interface.
@torch.no_grad()
def trace_hupp_op(A, nsamp):
    A_flat = A.flatten() # Flatten input and output shapes. 
    d = A_flat.shape_in[0]
    S = torch.randint(2, size=(d,nsamp//3)).float() * 2 - 1 # Either 1 or -1
    G = torch.randint(2, size=(d,nsamp//3)).float() * 2 - 1 # Either 1 or -1

    call = lambda x: A_flat.batched_call(x.T).T

    Q, _ = torch.linalg.qr(call(S))
    prod = G - Q @ (Q.T @ G)
    return (torch.trace(call(Q).T @ Q) + (3./nsamp) * torch.trace(call(prod).T @ prod)).item()

# Adjoint only hutch++.
@torch.no_grad()
def trace_hupp_adj_only(A, B, nsamp):
    # Assume A = B B^T.
    d = A.shape[0]
    S = np.random.randint(2, size=(d,nsamp)).astype(float) * 2 - 1 # Either 1 or -1
    G = np.random.randint(2, size=(d,nsamp)).astype(float) * 2 - 1 # Either 1 or -1

    Q, _ = np.linalg.qr(A @ S)
    prod = G - Q @ (Q.T @ G)
    return np.trace((B.T @ Q).T @ (B.T @ Q)) + (1./nsamp) * np.trace((B.T @ prod).T @ (B.T @ prod))

# Efficient cosine alignment re-using redundancies.
@torch.no_grad()
def op_alignment(A, B, nsamp=20, full_output=False):
    A_flat = A.flatten() # Flatten input and output shapes. 
    B_flat = B.flatten() 
    d = A_flat.shape_in[0]
    G = torch.randn(d, nsamp)
    Q, _ = torch.linalg.qr(G) # Random orthonormal vectors.
    S = (d / nsamp)**0.5 * Q          
    AS = A_flat.batched_call(S.T).T # This is BY FAR the bottle-neck.
    BS = B_flat.batched_call(S.T).T

    # Cumsum will return cumulative estimate up to sample count (to check if its converging).
    reduction = torch.sum
    if full_output:
        reduction = lambda x: torch.cumsum(torch.sum(x, 0), 0) 
    return (reduction(AS * BS) / (reduction(AS * AS) * reduction(BS * BS))**0.5).item()

@torch.no_grad()
def op_alignment_adj_only(J_1, J_2, K_1, K_2, nsamp=20, full_output=False):
    J_1_flat, J_2_flat = J_1.flatten(), J_2.flatten() # Flatten input and output shapes. 
    K_1_flat, K_2_flat = K_1.flatten(), K_2.flatten() 
    d = K_1_flat.shape_in # All should be square!

    # Setup stencils (two probes in this case).
    Gx, Gy = torch.randn(d, nsamp), torch.randn(d, nsamp)
    Qx, _ = torch.linalg.qr(Gx) # Random orthonormal vectors.
    Qy, _ = torch.linalg.qr(Gy)
    Sx, Sy = d**0.5 * Qx, d**0.5 * Qy

    # Evaluations (4*nsamp):
    J_1x = J_1_flat.batched_adjoint_call(Sx.T).T
    J_2y = J_2_flat.batched_adjoint_call(Sy.T).T
    K_1y = K_1_flat.batched_adjoint_call(Sy.T).T
    K_2x = K_2_flat.batched_adjoint_call(Sx.T).T # shape (d, nsamp)
    J_sum, K_sum = torch.sum(J_1x*J_2y, 0), torch.sum(K_1y*K_2x, 0) # shape (nsamp,)

    cum = []
    for n in range(2,nsamp+1):
        cum.append((J_sum*K_sum)[:n].mean() / ((J_sum[:n]**2).mean() * (K_sum[:n]**2).mean())**0.5)
    return torch.stack(cum) if full_output else cum[-1]

@torch.no_grad()
def op_alignment_variant_1(J_1, J_2, B, nsamp=20, full_output=False):
    # Assume A = J_1 J_2* and can only call adjoint rmatvec on J_1, J_2. 
    J_1_flat, J_2_flat = J_1.flatten(), J_2.flatten() # Flatten input and output shapes. 
    B_flat = B.flatten()

    # Setup stencils (two probes in this case).
    d = B_flat.shape_in
    G = torch.randn(d, nsamp).to(B.dev)
    Q, _ = torch.linalg.qr(G) # Random orthonormal vectors.
    S = d**0.5 * Q          

    # matvec evaluations. Main bottle-neck.
    J1_S = J_1_flat.batched_adjoint_call(S.T).T
    B_S = B_flat.batched_call(S.T).T
    J2_B_S = J_2_flat.batched_adjoint_call(B_S.T).T

    J1_S_sq = (J1_S ** 2).sum(0)

    def estimate(n):
        centered_sq = ((J1_S_sq[:n] - J1_S_sq[:n].mean())**2).sum()
        correction = (J1_S_sq**2).mean()
        trace_J1_moment = (1. / d) * correction + .5 * centered_sq / (n-1)
        trace_BB = (B_S[:,:n]**2).mean(1).sum()
        cross_trace = (J1_S * J2_B_S).mean(1).sum()
        return cross_trace / (trace_J1_moment * trace_BB)**0.5

    if full_output:
        return torch.stack([estimate(n) for n in range(2,nsamp+1)])
    return estimate(nsamp)

@torch.no_grad()
def op_alignment_variant_2(J_1, J_2, B, nsamp=20, full_output=False):
    # Assume A = J_1 J_2* and can only call adjoint rmatvec on J_1, J_2. 
    J_1_flat, J_2_flat = J_1.flatten(), J_2.flatten() # Flatten input and output shapes. 
    B_flat = B.flatten()

    # Setup stencils (two probes in this case).
    d = B_flat.shape_in
    G = torch.randn(d, nsamp).to(B.dev)
    Q, _ = torch.linalg.qr(G) # Random orthonormal vectors.
    S = d**0.5 * Q          

    # matvec evaluations. Main bottle-neck.
    J1_S = J_1_flat.batched_adjoint_call(S.T).T
    B_S = B_flat.batched_call(S.T).T
    J2_B_S = J_2_flat.batched_adjoint_call(B_S.T).T

    cum = []
    for n in range(2,nsamp+1):
        cross_trace = ((1/nsamp) * J1_S * J2_B_S).sum()
        trace_BB = ((1/nsamp) * B_S[:,:n]**2).sum()
        trace_JJ = ((1/nsamp) * J1_S[:,:n]**4).sum()
        cum.append(cross_trace / (trace_JJ * trace_BB)**0.5)

    return torch.stack(cum) if full_output else cum[-1]

@torch.no_grad()
def trace_JJtJJt(J, nsamp=20, full_output=False):
    J_flat = J.flatten() # Flatten input and output shapes. 
    d = J_flat.shape_out
    G = torch.randn(d, nsamp)
    Q, _ = torch.linalg.qr(G) # Random orthonormal vectors.
    S = d**0.5 * Q          

    s = (J_flat.batched_adjoint_call(S.T).T**2).sum(0)

    cum = []
    for n in range(2,nsamp+1):
#        cum.append((s[:n]**2).sum() / n)
#        cum.append((n * (s[:n]**2).sum() - s[:n].sum()**2) / (2 * (n-1) * n))
        centered_sq = ((s[:n] - s[:n].mean())**2).sum()
        cum.append(centered_sq / (2 * (n-1)))

    return torch.stack(cum) if full_output else cum[-1]
