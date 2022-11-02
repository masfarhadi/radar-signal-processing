# -*- coding: utf-8 -*-
"""
Radar signal processing homework
The Fast Fourier Transform
RaSigUE_slides_5

@author: Mas.farhadi
"""

import numpy as np
import scipy.special
import scipy.signal as sp
# import scipy.stats
import matplotlib.pyplot as plt
from tikzplotlib import save as tikz_save
import spectrum

# generate the samples
wind_samples = spectrum.window.window_nuttall(128)
# generate window object
for name in ('hanning', 'nuttall'):
    wind = spectrum.Window(128, name)      # generate a window object
    wind_samples = wind.data              # get the data
    plt.figure()
    wind.plot_time_freq()               # plot the window function
    tikz_save('Window_'+name+'.tikz');
    plt.savefig('Window_'+name+'.png', dpi=150)


wind = np.hanning(128)
wind = sp.windows.nuttall(128)
wind = sp.get_window('hamming', 128)

# %%---- Visualization of the FFT Spectrum ----
N = 32
n_smaples = np.arange(N)
fs = 5*1e6  # sampling frequency is 5 MHz
Ts = 1 / fs
amp = 2
psi = 0.125  # 0.14  #
phi = 0
n_fft = N
freq = np.linspace(0, fs / 1e6, n_fft, endpoint=False)
freq = np.linspace(0, 1, n_fft, endpoint=False)

x = amp * np.exp(1j * 2 * np.pi * psi * n_smaples + phi)
x_fft = np.fft.fft(x, n_fft)

plt.figure('Visualization_FFT_Spectrum')
plt.title('Visualization_FFT_Spectrum')
plt.plot(freq, 20 * np.log10(1 / n_fft * np.abs(x_fft)), '-x', label='PSD', )
# plt.plot(fs*psi/1e6, 20*np.log10(np.abs(amp)), 'o', markersize=8, label='parameter',)
plt.plot(psi, 20*np.log10(np.abs(amp)), 'o', markersize=8, label='parameter',)
# plt.xlabel('Frequency (MHz)')
plt.xlabel('Normalized Frequency')
plt.ylabel('Power (dB)')
plt.legend()
plt.grid()
plt.show()

psi = 0.125
x = amp * np.exp(1j * 2 * np.pi * psi * n_smaples + phi)
x_fft = np.fft.fft(x, n_fft)

plt.figure('Visualization_FFT_Spectrum')
plt.title('Visualization_FFT_Spectrum')
plt.plot(freq, 20 * np.log10(1 / n_fft * np.abs(x_fft)), '-x', label='PSD', )
plt.plot(fs*psi/1e6, 20*np.log10(np.abs(amp)), 'o', markersize=6, label='parameter',)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Power (dB)')
plt.legend()
plt.grid()
plt.show()

