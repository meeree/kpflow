import torch

def neumann(A, b, max_iter=1000, tol=1e-8, relative=True, early_stop = True):
    """
    A^{-1} b = sum_k (I - A)^k b.
    """

    x = b.clone()
    term = b.clone()

    norms = []
    for itr in range(max_iter):
        term = term - A(term)   # B term = (I - A) term
        x = x + term

        threshold = tol * x.norm() if relative else tol
        norms.append(term.norm())
        if early_stop:
            if term.norm() <= threshold:
                break
    
    if itr == max_iter:
        print(f"Error: Neumann solve did not converge in {max_iter} Iterrations. Either A is not nilpotent and rho(A) >= 1, or need more iters.")
    return x, itr
