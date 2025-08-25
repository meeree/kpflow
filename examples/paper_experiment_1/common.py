def project(data):
    from sklearn.decomposition import PCA
    data_flat = data.reshape((-1, data.shape[-1]))
    pca = PCA().fit(data_flat) 
    return pca, pca.transform(data_flat).reshape(data.shape)


def set_mpl_defaults(fontsize = 13): # Fontsize etc
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    plt.rc('font', size=fontsize)          # controls default text sizes
    plt.rc('axes', titlesize=fontsize)     # fontsize of the axes title
    plt.rc('axes', labelsize=fontsize)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsize)    # fontsize of the tick labels
    plt.rc('legend', fontsize=fontsize)    # legend fontsize
    plt.rc('figure', titlesize=fontsize)  # fontsize of the figure title
    mpl.rcParams['mathtext.fontset']   = 'cm'        # use Computer Modern
    mpl.rcParams['font.family']        = 'serif'     # make non‑math text serif

def plot_trajectories(data, m = 1, n = 1, i = 1, dim = 3, legend = True, colors = None):
    # data should be shape [batch count, time, hidden dim].
    import matplotlib.pyplot as plt
    PALLETTE = ['#f86969ff', '#7e69f8ff', '#f8c969ff', '#69f87cff', '#e569f8ff']
    plt.subplot(m, n, i, projection = None if dim == 2 else '3d')
    for idx, traj in enumerate(data):
        if dim == 3:
            if colors is not None:
                plt.plot(traj[:, 0], traj[:, 1], traj[:, 2], color = colors[idx])
            else:
                plt.plot(traj[:31, 0], traj[:31, 1], traj[:31, 2], color = PALLETTE[0])
                plt.plot(traj[30:61, 0], traj[30:61, 1], traj[30:61, 2], color = PALLETTE[1])
                plt.plot(traj[60:, 0], traj[60:, 1], traj[60:, 2], color = PALLETTE[2])
        else:
            if colors is not None:
                plt.plot(traj[:, 0], traj[:, 1], color = colors[idx])
            else:
                plt.plot(traj[:31, 0], traj[:31, 1], color = PALLETTE[0])
                plt.plot(traj[30:61, 0], traj[30:61, 1], color = PALLETTE[1])
                plt.plot(traj[60:, 0], traj[60:, 1], color = PALLETTE[2])
    if legend:
        plt.legend(['stim', 'mem', 'resp'])

def compute_svs(op, inp_shape, ncomps, compute_vecs = False, tol = 1e-8):
    from scipy.sparse.linalg import eigsh
    op_sp = op.to_scipy(inp_shape, inp_shape, dtype = float, can_matmat = False)
    if compute_vecs:
        singular_vals, singular_vecs = eigsh(op_sp, k = ncomps, return_eigenvectors = True, tol = tol)
        return singular_vals[::-1], singular_vecs[:, ::-1].T.reshape((-1, *inp_shape))

    singular_vals = eigsh(op_sp, k = ncomps, return_eigenvectors = False, tol = tol)
    return singular_vals[::-1]
