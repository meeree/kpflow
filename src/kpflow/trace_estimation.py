import numpy as np
import matplotlib.pyplot as plt

# Hutch++ Algorithm.
def trace_hupp(A, nsamp):
    d = A.shape[0]
    S = np.random.randint(2, size=(d,nsamp)).astype(float) * 2 - 1 # Either 1 or -1
    G = np.random.randint(2, size=(d,nsamp)).astype(float) * 2 - 1 # Either 1 or -1

    Q, _ = np.linalg.qr(A @ S)
    prod = G - Q @ (Q.T @ G)
    return np.trace((A @ Q).T @ Q) + (1./nsamp) * np.trace((A @ prod).T @ prod)

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


#nsamps = (10**np.linspace(1, 3, 10)).astype(int)
#
#d = 10000
#A = np.zeros((d,d))
#
#plt.figure(figsize = (4 * 3, 3))
#for idx, k in enumerate([3, 100, 1000]):
#    U = np.random.normal(size=(d,k))
#    V = np.random.normal(size=(d,k))
#    A = U @ V.T
#
#    guesses = []
#    for nsamp in nsamps:
#        guesses.append(trace_hpp(A, nsamp))
#
#    plt.subplot(1,3,1+idx)
#    plt.plot(nsamps, guesses, color = 'blue')
#    plt.axhline(np.trace(A), color = 'red', linestyle = 'dashed')
#
#plt.show()
