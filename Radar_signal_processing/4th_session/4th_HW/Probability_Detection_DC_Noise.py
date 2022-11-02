# -*- coding: utf-8 -*-
"""
Radar signal processing homework
Probability of Detection: DC in Noise
RaSigUE_slides_4

@author: Mas.farhadi
"""

import numpy as np
import scipy.special
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save


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
noise = np.random.normal(mu, sigma, n_samples) + 1j * np.random.normal(mu, sigma, n_samples)
a_values = np.linspace(0.1, 15)
detection = np.zeros(len(a_values))
false_alarm = np.zeros(len(a_values))
snr = np.zeros(len(a_values))

detection_rice = np.zeros(len(a_values))
falarm_rayleigh = np.zeros(len(a_values))
for cnt, a in enumerate(a_values):
    signal = a + noise
    vt = a  # a/2  # a*2
    detection[cnt] = np.sum(abs(signal) > vt)/n_samples
    # pdf_rice = st.rice.pdf(x, a / sigma, scale=sigma)
    detection_rice[cnt] = 1 - st.rice.cdf(vt, a / sigma, scale=sigma)
    false_alarm[cnt] = np.sum(abs(noise) > vt)/n_samples
    falarm_rayleigh[cnt] = 1 - st.rayleigh.cdf(vt, scale=sigma)
    snr[cnt] = (a ** 2) / (2 * sigma ** 2)

plt.figure('Detection_probability')
plt.title('Detection_probability')
plt.plot(10*np.log10(snr), detection, label='detection')
# plt.plot(10*np.log10(snr), detection_rice, label='detection_rice')
plt.plot(10*np.log10(snr), false_alarm, label='false_alarm')
# plt.plot(10*np.log10(snr), falarm_rayleigh, label='false_alarm_rayleigh')
plt.grid()
plt.legend()
plt.xlabel('snr (dB)')
plt.ylabel('normalized rate')

plt.figure('Detection_probability_with_distribution')
plt.title('Detection_probability_with_distribution')
# plt.plot(10*np.log10(snr), detection, label='detection')
plt.plot(10 * np.log10(snr), detection_rice, label='detection_rice')
# plt.plot(10*np.log10(snr), false_alarm, label='false_alarm')
plt.plot(10 * np.log10(snr), falarm_rayleigh, label='false_alarm_rayleigh')
plt.grid()
plt.legend()
plt.xlabel('snr (dB)')
plt.ylabel('normalized rate')



