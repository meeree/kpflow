
import numpy as np

def project(data):
    from sklearn.decomposition import PCA
    data_flat = data.reshape((-1, data.shape[-1]))
    ncomp = min(data_flat.shape)
    pca = PCA(ncomp).fit(data_flat) 
    return pca, pca.transform(data_flat).reshape((*data.shape[:-1], ncomp))

def effdim(data_, center = True):
    data = torch_to_np(data_)
    if center: # PCA (center the data first):
        pca, proj = project(torch_to_np(data))
        r = pca.explained_variance_ratio_
        return 1.0 / np.sum(r**2)
#        return np.argmax(np.cumsum(pca.explained_variance_ratio_) > .95) + 1 # Explain 95% of variance.
    # SVD (no shift):
    data_flat = data.reshape((-1, data.shape[-1]))
    mat = data_flat.T @ data_flat / data_flat.shape[0]
    return np.trace(mat)**2 / np.trace(mat @ mat)

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
    import torch
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x 

def np_to_torch(x, dev = 'cpu'):
    import torch
    return x if torch.is_tensor(x) else torch.from_numpy(x).to(dev)

def skree_plot(pca, m = 1, n = 1, i = 1):
    import matplotlib.pyplot as plt
    plt.subplot(m, n, i)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))

def set_mpl_defaults(fontsize = 13): # Fontsize etc
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    plt.rc('font', size=fontsize)          # controls default text sizes
    plt.rc('axes', titlesize=fontsize)     # fontsize of the axes title
    plt.rc('axes', labelsize=fontsize)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=fontsize-1)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsize-1)    # fontsize of the tick labels
    plt.rc('legend', fontsize=fontsize-1)    # legend fontsize
    plt.rc('figure', titlesize=fontsize+1)  # fontsize of the figure title
    mpl.rcParams['mathtext.fontset']   = 'cm'        # use Computer Modern
    mpl.rcParams['font.family']        = 'serif'     # make non‑math text serif
    mpl.rcParams['font.serif']         = 'DejaVu Serif' #'Computer Modern Roman'
    mpl.rcParams['pdf.fonttype'] = 42   # TrueType
    mpl.rcParams['ps.fonttype'] = 42


def plot_trajectories(data, m = 1, n = 1, i = 1, dim = 3, legend = True, c = None, cmap = None):
    # data should be shape [batch count, time, hidden dim].
    import matplotlib.pyplot as plt
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    plt.subplot(m, n, i, projection = None if dim == 2 else '3d')
    for idx, traj in enumerate(data):
        extra = {}
        if c is not None:
            extra['color'] = c[idx] if cmap is None else cmap((c[idx] - c.min()) / (c.max() - c.min()))

        if dim == 3:
            plt.plot(traj[:, 0], traj[:, 1], traj[:, 2], **extra)
            plt.gca().set_zlabel('PC3')
        else:
            plt.plot(traj[:, 0], traj[:, 1], **extra)
        plt.xlabel('PC1')
        plt.ylabel('PC2')

def plot_traj_mempro(ang, data, m = 1, n = 1, i = 1, dim = 3, legend = True, colors = None):
    # data should be shape [batch count, time, hidden dim].
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    PALLETTE = ['#f86969ff', '#7e69f8ff', '#f8c969ff', '#69f87cff', '#e569f8ff']
    plt.subplot(m, n, i, projection = None if dim == 2 else '3d')
    for idx, (traj, my_ang) in enumerate(zip(data, ang)):
        t = (my_ang + np.pi) / (2 * np.pi) * .4 + .6
        scale = lambda c : np.array(to_rgba(c)) * t
        if dim == 3:
            plt.plot(traj[:31, 0], traj[:31, 1], traj[:31, 2], color = scale(PALLETTE[0]))
            plt.plot(traj[30:61, 0], traj[30:61, 1], traj[30:61, 2], color = scale(PALLETTE[1]))
            plt.plot(traj[60:, 0], traj[60:, 1], traj[60:, 2], color = scale(PALLETTE[2]))
        else:
            plt.plot(traj[:31, 0], traj[:31, 1], color = scale(PALLETTE[0]))
            plt.plot(traj[30:61, 0], traj[30:61, 1], color = scale(PALLETTE[1]))
            plt.plot(traj[60:, 0], traj[60:, 1], color = scale(PALLETTE[2]))
    if legend:
        plt.legend(['stim', 'mem', 'resp'])

def compute_svs(op, ncomps, compute_vecs = False, tol = 1e-8):
    from scipy.sparse.linalg import eigsh
    op_sp = op.to_scipy()
    if compute_vecs:
        singular_vals, singular_vecs = eigsh(op_sp, k = ncomps, return_eigenvectors = True, tol = tol)
        return singular_vals[::-1], singular_vecs[:, ::-1].T.reshape((-1, *inp_shape))

    singular_vals = eigsh(op_sp, k = ncomps, return_eigenvectors = False, tol = tol)
    return singular_vals[::-1]

def imshow_nonuniform(X, Y, Z, nx=100, ny=100, **kwargs):
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    xi = np.linspace(X.min(), X.max(), nx)
    yi = np.linspace(Y.min(), Y.max(), ny)
    print(X.min(), Y.min(), X.max(), Y.max())
    Xi, Yi = np.meshgrid(xi, yi)
    to_grid = lambda z: griddata((X.ravel(), Y.ravel()), z.ravel(), (Xi, Yi), method='linear')
    if len(Z.shape) > 1: # RGBA colors
        Zi = np.stack([to_grid(z) for z in Z.T], -1)
    else:
        Zi = to_grid(Z)

    plt.imshow(Zi, origin = 'lower', extent = [xi.min(), xi.max(), yi.min(), yi.max()], **kwargs)
    return Xi, Yi, Zi

def absolute_error(x, y):
    return np.abs(torch_to_np(x) - torch_to_np(y)).max()
def relative_error(x, y):
    return absolute_error(x, y) /  max(np.abs(torch_to_np(x)).max(), np.abs(torch_to_np(y)).max())


def annotate_subplots(x = -0.08, y = 1.18):
    """Annotate all axes in the current figure with bold A, B, C... in the top-left corner."""
    import matplotlib.pyplot as plt, string
    fig = plt.gcf()
    axes = fig.get_axes()

    for i, ax in enumerate(axes):
        if i >= len(string.ascii_uppercase):
            break  # stop after Z, we’re civilized
        label = string.ascii_uppercase[i]
        ax.text(
            x, y, label, transform=ax.transAxes,
            fontweight='bold',
            va='top', ha='left'
        )

def plot_err_bar(xdata, data, percentile_range, color = None, label = '', alpha = .5):
    import matplotlib.pyplot as plt

    # Assume samples are in in axis 0.
    median = np.nanmedian(data, axis=0)               
    qmin = np.percentile(data, 50 - percentile_range, axis=0)            
    qmax = np.percentile(data, 50 + percentile_range, axis=0)

    plt.plot(xdata, median, label = label, color = color)
    plt.fill_between(xdata, qmin, qmax, facecolor = color, label = '_nolabel_', alpha = .5)
