import numpy as np
import matplotlib.pyplot as plt
import torch

# Hutch++ Algorithm.
def trace_hupp(A, nsamp):
    d = A.shape[0]
    S = np.random.randint(2, size=(d,nsamp)).astype(float) * 2 - 1 # Either 1 or -1
    G = np.random.randint(2, size=(d,nsamp)).astype(float) * 2 - 1 # Either 1 or -1

    Q, _ = np.linalg.qr(A @ S)
    prod = G - Q @ (Q.T @ G)
    return np.trace((A @ Q).T @ Q) + (1./nsamp) * np.trace((A @ prod).T @ prod)

def trace_hupp_op(A, nsamp):
    A_flat = A.flatten() # Flatten input and output shapes. 
    d = A_flat.shape_in
    S = torch.randint(2, size=(d,nsamp)).float() * 2 - 1 # Either 1 or -1
    G = torch.randint(2, size=(d,nsamp)).float() * 2 - 1 # Either 1 or -1

    call = lambda x: A_flat.batched_call(x.T).T

    Q, _ = torch.linalg.qr(call(S))
    prod = G - Q @ (Q.T @ G)
    return torch.trace(call(Q).T @ Q) + (1./nsamp) * torch.trace(call(prod).T @ prod)

def trace_hupp_adj_only(A, B, nsamp):
    # Assume A = B B^T.
    print(B.T.shape)
    d = A.shape[0]
    S = np.random.randint(2, size=(d,nsamp)).astype(float) * 2 - 1 # Either 1 or -1
    G = np.random.randint(2, size=(d,nsamp)).astype(float) * 2 - 1 # Either 1 or -1

    Q, _ = np.linalg.qr(A @ S)
    prod = G - Q @ (Q.T @ G)
    return np.trace((B.T @ Q).T @ (B.T @ Q)) + (1./nsamp) * np.trace((B.T @ prod).T @ (B.T @ prod))

# Nystrom++ Algorithm.
def trace_nypp(A, nsamp):
    d = A.shape[0]
    S = np.random.randint(2, size=(d,nsamp)).astype(float) * 2 - 1 # Either 1 or -1
    G = np.random.randint(2, size=(d,nsamp)).astype(float) * 2 - 1 # Either 1 or -1

    Q, _ = np.linalg.qr(A @ S)
    prod = G - Q @ (Q.T @ G)
    return np.trace((A @ Q).T @ Q) + (1./nsamp) * np.trace((A @ prod).T @ prod)

def op_alignment(A, B, shape, nsamp):
    num = trace_hupp((A.T() @ B).to_scipy(shape), nsamp)
    denom = (trace_nypp((A.T() @ A).to_scipy(shape), nsamp) * trace_nypp((B.T() @ B).to_scipy(shape), nsamp))**0.5
    return num / denom
