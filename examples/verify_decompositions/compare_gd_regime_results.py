import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

from common import project, plot_trajectories, compute_svs, set_mpl_defaults, plot_traj_mempro, imshow_nonuniform, effdim, relative_error, plot_err_bar, skree_plot, annotate_subplots

losses = np.genfromtxt('loss_data.csv', delimiter=',', dtype = float)
set_mpl_defaults(14)

plt.figure(figsize = (4,3))
plt.plot(losses)
#plt.yscale('log')
plt.xlabel('GD Iteration')
plt.ylabel('Loss (mse)')
plt.legend(['Case I', 'Case II', 'Case III'], loc = 'upper right')
plt.tight_layout()
plt.savefig('loss_regimes.pdf')
plt.show()
