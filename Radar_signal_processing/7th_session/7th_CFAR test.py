# -*- coding: utf-8 -*-
"""
Radar signal processing homework
The CFAR test
RaSigUE_slides_7

@author: Mas.farhadi
"""

import numpy as np
import scipy.special
import scipy.signal as sp
# import scipy.stats
import matplotlib.pyplot as plt
from numba import jit
from matplotlib import axes
from matplotlib2tikz import save as tikz_save


# @jit(nopython=True, cache=True, parallel=True)
def cfar_thresh_lvl(x, n_width, Pfa=None, n_guard=None, mode=None):
    """
    Calculates the threshold level for a signal x with a CFAR
    method given by mode, by looping over all cells.
    Parameters
    ----------
    x: array
    Array of positive (absolute values) of floats
    n_width: int
    One-sided width of the window.
    Pfa: float, optional
    False alarm rate. Default: 1e-4
    n_guard: int, optional
    One sided number of guard cells. Default: no guard cells
    mode: string, optional
    'CA' or None for cell average, 'CAGO' for CA-greatest of
    Returns
    -------
    array of size of x holding threshold levels.
    """
    if Pfa is None:
        Pfa = 1e-2  # probability of false-alarm
    L = 2*n_width  # double of n_width
    alpha = np.sqrt(4/np.pi*L*(Pfa**(-1/L)-1)*(1-(1-np.pi/4)*np.exp(1-L)))

    # calculate threshold levels array
    thr_arr = np.zeros(x.size)
    for cnt in range(np.int(n_width+n_guard), np.int(x.size-(n_width+n_guard))):
        left_sum = np.sum(x[cnt-(n_width+n_guard):cnt - n_guard])
        right_sum = np.sum(x[cnt+n_guard:cnt+(n_guard + n_width)])
        if mode is None: mode = 'CA'
        if mode == 'CA':
            thr_arr[cnt] = np.sum(left_sum+right_sum)/L*alpha
        else:
            thr_arr[cnt] = 2*np.max([left_sum, right_sum])/L*alpha

    return thr_arr


#%% main
Pfa = 1e-2  # probability of false-alarm
n_width = 12  # single-sided width of the window
n_guard = 1  # single-sided number of guard cells

n = np.concatenate((np.random.randn(100), 5*np.random.rand(40)))
x = np.abs(n)  # Rayleigh distributed
x[[32,56,75,77,83,112]]=[7,4,12,8,10,10]  # setting multiple elements

CFAR_arr = np.zeros(x.size - 2 * (n_width + n_guard))  # CFAR values
res_arr = []  # the result indexes which is more than threshold
thr_limit = np.arange((n_width+n_guard),x.size-(n_width+n_guard))  # limit that we show the CFAR result

# CAGO calculation
cago_thrs = cfar_thresh_lvl(x, n_width, n_guard=n_guard, mode='CAGO')
CFAR_CAGO = np.where(x[thr_limit] > cago_thrs[thr_limit])+thr_limit[0]

# CA calculation
ca_thrs = cfar_thresh_lvl(x, n_width, n_guard=n_guard, mode='CA')
CFAR_CA = np.where(x[thr_limit] > ca_thrs[thr_limit])+thr_limit[0]

plt.figure()
plt.plot(x, label='signal', linewidth=2, )

plt.plot(thr_limit, ca_thrs[thr_limit], label='CA', linestyle='-.', color='g')
plt.plot(CFAR_CA, ca_thrs[CFAR_CA], 's', color='g', markersize=12)

plt.plot(thr_limit, cago_thrs[thr_limit], label='CAGO', linestyle='-.', color='r')
plt.plot(CFAR_CAGO, cago_thrs[CFAR_CAGO], 's', color='r', markersize=12)

plt.grid()
plt.legend()


# # add wrapper
# c = jit(cfar_thresh_lvl, nopython=True, cache=True)
# # compare result
# t1 = c(x, n_width, Pfa)  # call to wrapped function
# t2 = cfar_thresh_lvl(x, n_width, Pfa)  # call to original function
# np.allclose(t1, t2) # True

