import torch

def trace_direct(ntk, dev = 'cpu'): # Chunking vmaps instead of looping makes this code faster. Vanilla looping is insanely slow.
    A_flat = ntk.flatten()
    d = A_flat.shape_in[0]

    call = lambda x: A_flat.batched_call(x.T).T
    E = torch.eye(d)
    return torch.trace(call(E)).item()

def trace_hupp_op(ntk, nsamp):
    A_flat = ntk.flatten() # Flatten input and output shapes. 
    d = A_flat.shape_in[0]
    if nsamp > d:
        return trace_direct(ntk)

    q = nsamp // 6 # can be tweaked.
    s = (nsamp - q) // 2  # need 2 * s + q = nsamp.
    q = q + (nsamp - q) % 2 # In case nsamp - q is odd.

    S = torch.randint(2, size=(d,q)).float() * 2 - 1 # Either 1 or -1
    T = torch.randint(2, size=(d,s)).float() * 2 - 1 # Either 1 or -1

    call = lambda x: A_flat.batched_call(x.T).T

    Q, _ = torch.linalg.qr(call(S), mode='reduced')
    prod = T - Q @ (Q.T @ T)
    s = prod.shape[1] # min(s, d)
    return (torch.trace(call(Q).T @ Q) + (1/s) * torch.trace(call(prod).T @ prod)).item()
    
def trace_one_sided(one_side, nsamp, dev = 'cpu'):
    A_flat = one_side.flatten()
    d = A_flat.shape_in[0]
    call_single = lambda x: torch.linalg.norm(A_flat._matvec(x))**2
    call_nm = torch.vmap(call_single)

    q = nsamp // 6
    S = (torch.randint(2, size=(d, q), device=dev).float() * 2 - 1).to(torch.float64).to(dev)
    Q, _ = torch.linalg.qr(S, mode='reduced')
    k = Q.shape[1]

    tau_Q = torch.sum(call_nm(Q.T))

    if k >= d:
        return tau_Q.item()

    T = (torch.randint(2, size=(d, nsamp - q), device=dev).float() * 2 - 1).to(torch.float64).to(dev)
    PT = T - Q @ (Q.T @ T)
    s = PT.shape[1]

    tau_perp = (1.0 / s) * torch.sum(call_nm(PT.T))   # no (d-k) factor here
    return (tau_Q + tau_perp).item()
