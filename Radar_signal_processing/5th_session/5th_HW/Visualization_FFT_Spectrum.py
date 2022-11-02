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
from matplotlib2tikz import save as tikz_save
import spectrum

# %% ---- test ---- # generate the samples
wind_samples = spectrum.window.window_nuttall(128)
# generate window object
for name in ('rectangular', 'hanning', 'nuttall'):
    wind = spectrum.Window(128, name)      # generate a window object
    wind_samples = wind.data              # get the data
    plt.figure()
    wind.plot_time_freq()               # plot the window function
    tikz_save('Window_'+name+'.tikz')
    plt.savefig('Window_'+name+'.png', dpi=150)


wind = np.hanning(128)
wind = sp.windows.nuttall(128)
wind = sp.get_window('hamming', 128)

# %%---- Visualization of the FFT Spectrum ----
N = 128
n_smaples = np.arange(N)
fs = 2*1e6  # sampling frequency is 5 MHz
Ts = 1 / fs
amp = np.array([1.0, 0.5, 1e-4, 0.001]).reshape(-1, 1)
psim = np.array([0.1, 0.1+2.5/N, 0.17, 0.21]).reshape(-1, 1)
phim = np.array([0.0, 0.0, 2.0, 0.0]).reshape(-1, 1)

x = amp * np.sin(2 * np.pi * psim * n_smaples + phim)
N_fft = N
for wind_type in ['rect', 'hanning', 'nuttall']:
    wind = sp.get_window(wind_type, N)
    x_win = wind * x
    x_fft = np.fft.fft(x_win, N_fft, axis=1)
    freq = np.linspace(0, fs/1e6, N_fft, endpoint=False)

    plt.figure('Visualization_FFT_Spectrum ' + str(wind_type) + ' window')
    plt.title('Visualization_FFT_Spectrum ' + str(wind_type) + ' window')
    plt.plot(freq, 20*np.log10(1/N_fft*np.abs(x_fft.T)), '-x', label='PSD',)
    plt.plot(fs * psim / 1e6, 20 * np.log10(np.abs(amp)), 'o', markersize=6, label='parameters', )
    plt.xlabel('Frequency (MHz)')
    plt.xlim(0.1, 0.5)  # MHz
    plt.ylabel('Power (dB)')
    plt.ylim(-100, 10)  #  dB
    plt.legend(['PSD_1', 'PSD_2', 'PSD_3', 'PSD_4', 'parameters'])
    plt.grid()
    plt.show()
    tikz_save('HW_Window_'+wind_type+'.tikz')
    plt.savefig('HW_Window_'+wind_type+'.png', dpi=150)

