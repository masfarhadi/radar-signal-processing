# -*- coding: utf-8 -*-
"""
Radar signal processing homework
Cross correlation
page 14/14 of RaSigUE_slides_3

@author: Mas.farhadi
"""

import numpy as np
import scipy.special
# import scipy.stats
import matplotlib.pyplot as plt


def rect_func(x, max_value):
    return np.where(abs(x) <= max_value, 1, 0)


def barker_corr_func():
    barker_11 = np.array([+1, +1, +1, -1, -1, -1, +1, -1, -1, +1, -1])
    x_seq = np.arange(-5, 6)
    rect_11 = rect_func(x_seq, max_value=5)
    plt.figure('auto correlation ')
    plt.subplot(222)
    markerline1,_,_ = plt.stem(x_seq, rect_11, label = 'rect_signal',)
    plt.setp(markerline1, 'markerfacecolor', 'b')
    plt.title('rect')
    plt.xlim(-11, 11)
    plt.legend()
    plt.grid(True)

    plt.subplot(221)
    markerline1, _, _ =plt.stem(x_seq, barker_11, label='barker_11',linefmt='-.')
    plt.setp(markerline1, 'markerfacecolor', 'r')
    plt.legend()
    # plt.xlabel('sequence samples')
    plt.xlim(-11, 11)
    plt.ylabel('signal')
    plt.title('Barker11')
    plt.grid(True)

    y_seq = np.arange(-10, 11)
    barker_corr = np.correlate(barker_11, barker_11,'full')
    rect_corr = np.correlate(rect_11, rect_11,'full')
    plt.figure('auto correlation ')
    plt.subplot(224)
    markerline1,_,_ = plt.stem(y_seq, rect_corr, label = 'rect_corr',)
    plt.setp(markerline1, 'markerfacecolor', 'b')
    plt.legend()
    plt.xlim(-11, 11)
    plt.grid(True)
    plt.subplot(223)
    markerline1, _, _ =plt.stem(y_seq, barker_corr, label='barker_corr',linefmt='-.')
    plt.setp(markerline1, 'markerfacecolor', 'r')
    plt.legend()
    # plt.xlabel('sequence samples')
    plt.xlim(-11, 11)
    plt.ylabel('Auto-correlation values')
    plt.grid(True)
    plt.show()


def gauss_dist_func():
    N = 4096  # number of samples
    sigma = 2.0  # Note: it's the standard deviation here
    np.random.seed(0)  # any integer number will do
    z = sigma * np.random.randn(N)
    plt.hist(z, bins=int(np.sqrt(N)))
    plt.xlabel('Values (1)')
    plt.ylabel('Frequency (1)')

    np.random.randn()

    np.random.seed(0)  # this option to semi random
    sigma = 1.5
    mu = 0
    n_g = st.norm()
    x = np.linspace(st.norm.ppf(0.01), st.norm.ppf(0.99), 1000)
    ax.plot(x, norm.pdf(x), 'r-', lw=5, alpha=0.6, label='norm pdf')

    noise = sigma * (np.random.randn(1000) + 1j * np.random.randn(1000)) + mu
    pdf = st.norm.pdf(n_g)

    plt.figure()
    plt.hist(n_u, bins=bins, density=density, label='Normal', alpha=0.5)
    plt.hist(n_g, bins=bins, density=density, label='Uniform', zorder=-1)
    plt.grid();
    plt.legend();
    plt.title('Normalized Distributions for %d samples' % (N,))
    plt.xlabel('x (1)')
    plt.ylabel('Frequency of x (1)')

