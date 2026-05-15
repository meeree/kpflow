import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../')
from common import set_mpl_defaults
set_mpl_defaults(14)

Ts = np.arange(1, 21)[:,None] 
gs = np.array([0.1, 0.9, 1.1, 2])
W = gs ** Ts * (1 - gs ** (2 * (Ts[-1] - Ts))) / (1 - gs**2) # [time, g value]

plt.plot(Ts[:-1], W[:-1])
plt.legend([f'g = {g}' for g in gs])
plt.yscale('log')
plt.xlabel('Time, $t$')
plt.ylabel('Weight, $w_t$') 

plt.show()
