# -*- coding: utf-8 -*-
"""
Radar signal processing homework
Complex numbers
RaSigUE_slides_4

@author: Mas.farhadi
"""

import numpy as np
import scipy.special
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save

sig = np.zeros(1000, dtype=np.complex)
sig = np.exp(1j*2*np.pi*0.1*np.arange(1000))

j_value = np.sqrt(-1+0*1j)

sig_real = sig.real
sig_imag = sig.imag
sig_abs = np.abs(sig)
sig_angle = np.angle(sig)

# %% ---- Histograms using Matplotlib ----
np.random.seed(0)       # init pseudo-random number generator
N = 1000                  # number of samples
n_u = np.random.rand(N)   # normalized uniform distribution
n_g = np.random.randn(N)  # normalized Gaussian distribution
bins = int(np.sqrt(N))    # needs conversion to integer
for density in (False, True):
    plt.figure()
    plt.hist(n_u, bins=bins, density=density, label='Normal', alpha=0.5)
    plt.hist(n_g, bins=bins, density=density, label='Uniform', zorder=-1)
    plt.grid()
    plt.legend()
    plt.title('Normalized Distributions for %d samples' %(N,))
    plt.xlabel('x (1)')
    plt.ylabel('Frequency of x (1)')
    filename ='NormalizedDistributions_density'+str(density)
    tikz_save(filename+'.tikz')
    plt.savefig(filename+'.png', dpi=150)

# %%---- Histogram and PDF of Complex Gaussian noise ----
sigma = 1.5
mu = 0
n_samples = 10000
bins = int(np.sqrt(n_samples))
x = np.linspace(mu - 10*sigma, mu + 10*sigma, n_samples)
signal_gauss = np.random.normal(mu, sigma, n_samples) + 1j * np.random.normal(mu, sigma, n_samples)  # sigma * np.random.randn(1000)
pdf_gauss = st.norm.pdf(x, scale=sigma)
plt.figure('Histogram_complex_Gaussian')
plt.title('Histogram_complex_Gaussian')
plt.hist(signal_gauss.real, bins=bins, density=True, label='hist(I)', alpha=0.5, ec='b')
plt.hist(signal_gauss.imag, bins=bins, density=True, label='hist(Q)', alpha=0.5, ec='b')
plt.plot(x, pdf_gauss,  label='PDF Normal $N(0,\sigma^2)$', linewidth=3, color='g')
plt.grid()
plt.legend()
plt.xlim([-6, 6])
plt.xlabel('Values')
plt.ylabel('Normalized Frequency')


# %%---- Histogram and PDF of Rayleigh Distribution noise ----
amp_signal_gauss = np.abs(signal_gauss)
pdf_rayleigh = st.rayleigh.pdf(x, scale=sigma)
plt.figure('Histogram_magnitude_Rayleigh')
plt.title('Histogram_magnitude_Rayleigh')
plt.hist(amp_signal_gauss, bins=bins, density=True, label='hist(magnitude)', alpha=0.5, ec='b')
plt.plot(x, pdf_rayleigh, label='PDF rayleigh', linewidth=3, color='g')
plt.grid()
plt.legend()
plt.xlim([-1, 6])
plt.xlabel('Values')
plt.ylabel('Normalized Frequency')

# %%---- Histogram and PDF of Rice Distribution noise ----
a_dc = 8.0
rice_signal = a_dc + signal_gauss
rice_magnitude = np.abs(rice_signal)
plt.figure('Histogram_magnitude_Rice')
plt.title('Histogram_magnitude_Rice')
pdf_rice = st.rice.pdf(x, a_dc/sigma, scale=sigma)
plt.hist(rice_magnitude, bins=bins, density=True, label='hist(magnitude)', alpha=0.5, ec='b')
plt.plot(x, pdf_rice, label='PDF Rice', linewidth=3, color='g')
plt.grid()
plt.legend()
plt.xlim([0, 16])
plt.xlabel('Values')
plt.ylabel('Normalized Frequency')



