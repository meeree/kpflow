import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

from common import project, plot_trajectories, compute_svs, set_mpl_defaults, plot_traj_mempro, imshow_nonuniform, effdim, relative_error, plot_err_bar, skree_plot, annotate_subplots

#vecs = np.load('vecs_ntk.npy')
#Q = vecs.reshape((vecs.shape[0], -1)).T
#n, k = Q.shape
#
## Create more orthonormal modes
#m = 20
#perm = np.random.permutation(n)   # or np.arange(n) for deterministic
#Z = np.eye(n)[:, perm[:m]]
#Z = np.ones((n, k))
#Z = np.zeros((n, m))
#width = max(2, n // (2*m))
#for j in range(m):
#    c = int((j + 0.5) * n / m)
#    a = max(0, c - width)
#    b = min(n, c + width)
#    Z[a:b, j] = 1.0
#Z = Z - Q @ (Q.T @ Z)
#
## orthonormalize the complement
#Z, _ = np.linalg.qr(Z)
#vecs_ortho = Z.reshape(vecs.shape)
#
#plt.figure()
#for idx in range(20):
#    plt.subplot(4,5,idx+1)
#    mode = vecs_ortho[idx, :, :, 0]
#    plt.plot(mode.T)
#plt.show()
#
#



import numpy as np

def _orthonormalize(Q):
    # make columns orthonormal (safe even if already)
    return np.linalg.qr(np.asarray(Q, float), mode="reduced")[0]

def _project_out(Q, Z):
    # Z: (T, r), Q: (T, k) orthonormal
    return Z - Q @ (Q.T @ Z)

def _normalize_cols(Z, eps=1e-12):
    nrm = np.linalg.norm(Z, axis=0)
    keep = nrm > eps
    return Z[:, keep] / nrm[keep], keep

def _interp_score(v):
    # Heuristic “interpretability”: sparse-ish + smooth-ish + few sign flips
    # (tweak if you care more about one property)
    tv = np.sum(np.abs(np.diff(v)))                 # total variation (smaller = smoother)
    sparsity = np.linalg.norm(v, 1) / (np.linalg.norm(v, 2) + 1e-12)  # larger = more localized/sparse
    flips = np.sum(np.sign(v[1:]) != np.sign(v[:-1]))  # fewer = nicer
    return 1.5*sparsity - 0.02*tv - 0.2*flips

def make_interpretable_orthogonal_temporal_vectors(Q, num=6, *,
                                                   widths=(5, 10, 20, 40),
                                                   include_steps=True,
                                                   include_hats=True,
                                                   include_haar=True,
                                                   oversample=6,
                                                   seed=0):
    """
    Q: (T, k) temporal modes (can be piecewise, zeros, whatever).
    Returns V: (T, num) orthonormal vectors orthogonal to span(Q),
    chosen to look interpretable (localized/piecewise/smooth).
    """
    rng = np.random.default_rng(seed)
    Q = _orthonormalize(Q)
    T, k = Q.shape

    candidates = []

    # 1) Piecewise-constant interval indicators
    if include_steps:
        for w in widths:
            for start in range(0, T - w + 1, max(1, w // 2)):
                z = np.zeros(T)
                z[start:start+w] = 1.0
                candidates.append(z)

    # 2) Piecewise-linear "hat" bumps (compact support, very interpretable)
    if include_hats:
        for w in widths:
            for start in range(0, T - w + 1, max(1, w // 2)):
                mid = start + w // 2
                z = np.zeros(T)
                left = np.arange(start, mid)
                right = np.arange(mid, start + w)
                if len(left) > 0:
                    z[left] = (left - start) / max(1, (mid - start))
                if len(right) > 0:
                    z[right] = (start + w - right) / max(1, (start + w - mid))
                candidates.append(z)

    # 3) Coarse Haar-like wavelets (piecewise constant with +/- blocks)
    if include_haar:
        # dyadic-ish scales
        for w in widths:
            w = int(w)
            for start in range(0, T - 2*w + 1, max(1, w)):
                z = np.zeros(T)
                z[start:start+w] = 1.0
                z[start+w:start+2*w] = -1.0
                candidates.append(z)

    # 4) Optional: add a few random *structured* mixtures to help span
    # (still interpretable-ish because they’re combos of simple atoms)
    r = max(num * oversample, 1)
    if len(candidates) > 0:
        idx = rng.integers(0, len(candidates), size=(r, min(6, len(candidates))))
        for j in range(r):
            z = np.zeros(T)
            for i in idx[j]:
                z += candidates[i]
            candidates.append(z)

    Z = np.column_stack(candidates) if candidates else rng.standard_normal((T, r))

    # Project out Q and normalize
    Zp = _project_out(Q, Z)
    Zp, _ = _normalize_cols(Zp)

    # Score + greedy pick with orthogonalization to keep them distinct
    picked = []
    V = np.zeros((T, 0))

    scores = np.array([_interp_score(Zp[:, j]) for j in range(Zp.shape[1])])
    order = np.argsort(-scores)  # best first

    for j in order:
        v = Zp[:, j].copy()
        # keep new picks orthogonal to previous picks too
        if V.shape[1] > 0:
            v = v - V @ (V.T @ v)
        nrm = np.linalg.norm(v)
        if nrm < 1e-10:
            continue
        v /= nrm
        V = np.column_stack([V, v])
        picked.append(j)
        if V.shape[1] >= num:
            break

    return V  # columns orthonormal and orthogonal to Q

vecs = np.load('vecs_ntk.npy')
S = np.load('S_ntk.npy')
var = S**2
Q = vecs[:, 0, :, 0].reshape((vecs.shape[0], -1)).T
Q = vecs[:, :, :, 0].reshape((vecs.shape[0], -1)).T
V = make_interpretable_orthogonal_temporal_vectors(Q, num=21, widths=(8,16,32))
print(V.shape)

plt.figure()
colors = [*plt.rcParams['axes.prop_cycle'].by_key()['color']]
set_mpl_defaults(14)
label_lines = {}
plt.figure(figsize = (5, 4))
for idx in range(4):
#    plt.subplot(4,5,idx+1)
    mode = var[idx]*vecs[idx, :, :, 0]
    lines = plt.plot(mode.T, color = colors[idx], alpha = 0.5)
    label_lines[f'Mode {idx+1}'] = lines[0]
plt.legend(label_lines.values(), label_lines.keys(), loc = 'upper right')
plt.xlabel('Time')
plt.ylabel('Weighted Mode')
plt.tight_layout()
plt.savefig('weight_modes_ntk.pdf')

plt.figure()
task1 = vecs[0, :, :, 0] 
task1[:10] *= -.5
task1[10:] *= .3

task2 = np.zeros_like(task1)
task2[:, -20:] = 10 + np.arange(task2.shape[0])[:, None]
task2[5:, -20:] *= -3

label_lines = {}
plt.figure(figsize = (4, 6))
plt.subplot(5,1,(1,2))
lines = plt.plot(task1.T, color = 'green', alpha = 0.5)
plt.title('Preferred Task Target $y^*$')
plt.ylabel('Target Value')
plt.subplot(5,1,(4,5))
lines = plt.plot(task2.T, color = 'red', alpha = 0.5)
#    label_lines[f'Mode {idx+1}'] = lines[0]
#plt.legend(label_lines.values(), label_lines.keys(), loc = 'upper right')
plt.xlabel('Time')
plt.ylabel('Target Value')
plt.title('Nullspace Task Target $y^*$')
plt.tight_layout()
plt.savefig('example_tasks_null.pdf')
plt.show()
plt.show()


plt.figure()
for idx in range(20):
    plt.subplot(4,5,idx+1)
    mode = V[:,idx].reshape(vecs.shape[1:-1])
    print(mode.shape)
    plt.plot(mode.T)
plt.show()
