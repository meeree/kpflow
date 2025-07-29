# Utils for analysis of checkpoints during training. 
import numpy as np
import glob, re, os
import torch
from tqdm import tqdm

def load_checkpoints(root):
    # Given a root directdirectoryoy, return a list of filenames corresponding to all checkpoints in that root directory. 
    # Also return iteration count for each file. Each file should be of the form root + checkpoints/checkpoint_<number>.pt.
    if len(root) > 0 and root[-1] != '/' and root[-1] != '\\':
        root = root + '/'

    checkpoints = glob.glob(root + 'checkpoints/*.pt')
    checkpoints = sorted(checkpoints, key=lambda x: int(re.findall(r'\d+', x)[-1]))
    iteration = [int(re.findall(r'\d+', file)[-1]) for file in checkpoints]
    checkpoints = [root + 'checkpoints/' + os.path.basename(p) for p in checkpoints]
    return checkpoints, iteration

def ping_dir(directory, clear = False):
    # Check if directory exists and make if not. If clear flag is True, clear any contents of the directory if it exists.
    import os
    if len(directory) == 0:
        return 

    if os.path.exists(directory):
        if clear:
            import shutil
            shutil.rmtree(directory)
            os.mkdir(directory)
    else:
        os.mkdir(directory)

def torch_to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x 

def np_to_torch(x, dev = 'cpu'):
    return x if torch.is_tensor(x) else torch.from_numpy(x).to(dev)

def load_sweep_checkpoints(root):
    # Similar to load_checkpoints but for a sweep, which consists of a grid of hyperparameters. 
    # Each sweep trial has checkpoints of the form grid_<trial_number>/checkpoints/checkpoint_<number>.pt.
    import json 
    if len(root) > 0 and root[-1] != '/' and root[-1] != '\\':
        root = root + '/'

    trials = glob.glob(root + 'grid_*/')
    trials = sorted(trials, key=lambda x: int(re.findall(r'\d+', x)[-1]))
    all_checkpoints, all_iteration = [], []
    for trial in trials:
        checkpoints, iteration = load_checkpoints(trial)
        all_checkpoints.append(checkpoints)
        all_iteration.append(iteration)
    with open(root + 'grid_manifest.json') as fin:
        manifest = json.load(fin)
    return manifest, all_checkpoints, all_iteration

def import_checkpoint(ch, device = 'cpu'):
    return torch.load(ch, map_location = device, weights_only = True)

def rerun_trials(X, Y, checkpoints, model, compute_adj = False, device = 'cuda', verbose = True):
    # #####################################################################################################
    # Given a list of checkpoints, rerun on the same consistent data and possibly compute adjoints, etc.  |
    # Checkpoints can either be a list of file names or a list of pytorch state_dicts.                    |
    # X Should be shape [trials, timesteps, n_in], Y shape [trials, timesteps, n_out].                    |
    # #####################################################################################################
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    if isinstance(Y, np.ndarray):
        Y = torch.from_numpy(Y)

    inp = X.to(device).float()
    if compute_adj:
        targ = Y.to(device).float()

    n_in, n_out = X.shape[-1], Y.shape[-1]

    zs_all, adjs_all, out_all, losses_all = [], [], [], []
    grads_all = []
    to_np = lambda x: x.detach().cpu().numpy()

    if verbose:
        print("Re-evaluating on the Same Data.")
    pbar = tqdm(list(checkpoints)) if verbose else list(checkpoints)
    for ch in pbar:
        if isinstance(ch, str):
            if verbose:
                pbar.set_description(ch)
            state_dict = torch.load(ch, map_location = device, weights_only=True)['model']
        else:
            state_dict = ch # Assume ch IS just a state dict, not a str indicating where it should be.
        n_hidden = state_dict['W.weight'].shape[0]

        with torch.set_grad_enabled(compute_adj):
            model.load_state_dict(state_dict)
            model = model.to(device)

        if not compute_adj:
            zs_all.append(to_np(model(inp)[1]))
            continue # Done in this case.

        zs, adj, out, loss_unreduced, loss = model.analysis_mode(inp, targ)
        zs_all.append(to_np(zs)) # [B, T, H]
        adjs_all.append(to_np(adj)) # [B, T, H]
        out_all.append(to_np(out)) # [B, T, n_out]
        losses_all.append(to_np(loss_unreduced.mean(-1))) # [B, T].
        grads_all.append({key: to_np(param.grad) for key, param in model.named_parameters()})

    if not compute_adj:
        return np.stack(zs_all)
    return np.stack(zs_all), np.stack(adjs_all), np.stack(out_all), np.stack(losses_all), grads_all


def effective_rank(A, thresh):
    S = np.linalg.svd(A, full_matrices=False)[1]
    variances = S**2
    total_variance = np.sum(variances)

    # Find the smallest k such that cumsum_variances[k] >= threshold * total_variance
    return np.searchsorted(np.cumsum(variances), thresh * total_variance) + 1

def batched_cov_and_pcs(traj, traj2 = None, dim_thresh = 0.95, abs_thresh = 1e-6, normalize = True):
    # Get the covariance and principle components for data over checkpoints (D), batches (B), time (T), with hidden dimension (H). 
    # traj is shape [D, B, T, H1]. D is trials, B is baches (what we mean over), T is time, H1 is hidden index.
    # If traj2 is not None, we take cross covariances, assuming traj2 has the same shape. Traj2 shape [D, B, T, H2].
    D, B, T, H = traj.shape
    if normalize: # Normalization is useful if the data scale is very small, causing issues with abs_thresh. 
        traj = traj / np.mean(np.abs(traj))
        if traj2 is not None:
            traj2 = traj2 / np.mean(np.abs(traj2))

    covs = []
    if traj2 is None:
        H2 = traj.shape[-1]
        for zs in traj:
            for z_t in zs.transpose(1,2,0): # Shapes [H, H], [H, B]
                covs.append(np.cov(z_t))
    else:
        H2 = traj2.shape[-1]
        centered, centered2 = traj - traj.mean(1)[:, None], traj2 - traj2.mean(1)[:, None]
        for zs, zs2 in zip(centered, centered2):
            print(zs.shape)
            for z_t, z_t2 in zip(zs.transpose(1,2,0), zs2.transpose(1,2,0)):
                covs.append(np.dot(z_t, z_t2.T) / (z_t.shape[1] - 1))

    covs = np.stack(covs).reshape((D, T, H, H2))

    pcs, S, _ = np.linalg.svd(covs, full_matrices=False)
    variances = S**2
    total_variances = np.sum(variances, axis = -1)
    variance_ratios = np.cumsum(variances / total_variances[..., None], axis = -1)
    dimensions = np.zeros_like(total_variances)
    for i1 in range(variance_ratios.shape[0]):
        for i2 in range(variance_ratios.shape[1]):
            if total_variances[i1,i2] < abs_thresh:
                continue
            dim_idx = np.argwhere(variance_ratios[i1, i2] > dim_thresh)[0,0]
            v0 = 0. if dim_idx == 0 else variance_ratios[i1, i2, dim_idx-1]
            v1 = variance_ratios[i1, i2, dim_idx]
            dimensions[i1, i2] = dim_idx + (dim_thresh - v0) / (v1 - v0) + 1

    dimensions[total_variances < abs_thresh] = 0 # If the total variance is super low, the covariance is a point, so it doesn't make sense to use variance ratios.
    return covs, variances, pcs, variance_ratios, dimensions
