import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp



###############################################
N = 128
fs = 2*10**6
freq = np.linspace(0, fs/(10**6), N)
Am = np.array([1.0, 0.5, 1e-4, 0.001])
psim = np.array([0.1, 0.1+2.5/N, 0.17, 0.21])
phim = np.array([0.0, 0.0, 2.0, 0.0])

t = np.arange(N).reshape(-1, 1)
sigs = Am * np.sin(2 * np.pi * psim * t + phim)

sigs_fft = np.fft.fft(sigs, N, axis=0)
plt.figure('fft')
plt.plot(freq, 20*np.log10(1/N*np.abs(sigs_fft)))
#plt.plot(psim, 20*np.log10(1/N*np.abs(sigs_fft)), 'x', label='points')
plt.xlabel('dB')
plt.ylabel('Frequency MHz')
#plt.xlim([])
plt.grid()
plt.show()
