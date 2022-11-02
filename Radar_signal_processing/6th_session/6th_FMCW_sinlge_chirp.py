# -*- coding: utf-8 -*-
"""
Radar signal processing homework
The FMCW single chirp
RaSigUE_slides_6

@author: Mas.farhadi
"""

import numpy as np
import scipy.special
import scipy.signal as sp
# import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import axes
from matplotlib2tikz import save as tikz_save
import spectrum

plt.rcParams['axes.formatter.limits'] = [-4, 4]

c0 = 3*1e8  # speed of light
fs = 1e6        # IF sample frequency
Ts = 1/fs  # sampling interval
B = 250e6       # RF bandwidth
fc = 24.125e9   # carrier frequency
T = 1e-3        # chirp duration
k_rampe = B/T  # ramp slope
F_dB = 12       # noise figure of the system
T0 = 270        # system temperature
R = 50          # reference impedance
k_b = 1.38064852*1e-23  # Boltzmann constant
range_bin = c0/(2*B)

# Two close (spurious) targets and three wanted targets:
A0_arr = (np.array([20, 18, 0.1, 2, 0.1])*1e-6).reshape(-1, 1)  # magnitudes
r0_arr = np.array([0.001, 0.1, 15, 40, 80]) .reshape(-1, 1)  # ranges
v0_arr = np.array([0, 0, 1, 0, 28]).reshape(-1, 1)  # range-rates

N = int(T/Ts)
n = np.arange(N)  # np.linspace(0, T-Ts, int(T/Ts))
phi_m = 2*fc*r0_arr/c0 + n*Ts*(2*k_rampe*r0_arr + 2*fc*v0_arr)/c0 + ((n*Ts)**2)*(2*k_rampe*v0_arr)/c0
Te = T0*(10**(F_dB/10) - 1)*20
p_n = k_b*Te/T
v_n = np.sqrt(p_n*R)
w_noise = v_n*np.random.randn(N)
s_if = np.sum(A0_arr * np.cos(2*np.pi*phi_m), axis=0) + w_noise

plt.figure('single_chirp')
plt.title('single_chirp')
plt.plot(n/1000, s_if, label='$S_{IF}$', )
#plt.plot(fs*psi/1e6, 20*np.log10(np.abs(amp)), 'o', markersize=6, label='parameter',)
plt.xlabel('time (ms)')
plt.ylabel('amplitude (v)')
plt.legend()
plt.grid()


for wind_type in ['rect', 'hamming', 'nuttall']:
    wind = sp.get_window(wind_type, N)
    x_win = wind * s_if
    N_fft = 2**20
    x_fft = np.fft.fft(x_win, N_fft)
    freq = np.linspace(0, fs/1e6, N_fft, endpoint=False)
    plt.figure('FFT_Spectrum ')
    plt.title('Visualization_FFT_Spectrum')
    plt.ylim([-190, -90])
    plt.plot(freq[:int(N_fft/6)]*N*range_bin, 20*np.log10(1/N*np.abs(x_fft[:int(N_fft/6)])), label=str(wind_type) + ' window')


plt.plot(r0_arr, 20*np.log10(np.abs(A0_arr)), 'x', markersize=6, label='True_range',)
plt.hlines(-170, 0, 100,  linestyles='dashed', label='Noise Floor')
plt.xlabel('Range (m)')
# plt.xlim(0.1, 0.5)  # MHz
plt.ylabel('Power (dB)')
#plt.ylim(-100, 10)  #  dB
plt.legend()#['PSD_1', 'PSD_2', 'PSD_3', 'PSD_4', 'parameters'])
plt.grid()
plt.show()

# %% ---- test ----
a = np.array(range(3)) # a.shape==(3,);  len(a)==3; a.size==3; a.ndim==1
b = np.zeros((3,))     # b.shape==(3,);  len(b)==3; b.size==3; b.ndim==1
c = np.zeros((3,1))    # c.shape==(3,1); len(c)==3; c.size==3; c.ndim==2
d = np.zeros((1,3))    # d.shape==(1,3); len(d)==1; d.size==3; d.ndim==2

r = np.random.rand(4, 4)
r[:, 3].shape  # returns (4,)

r[0, 0] = 1.0
r[:, 0] = [3, 2, 1, 0]

a = np.arange(3)  # produces a 0D-array
b = a.T  # b is of same shape as a
c = a[np.newaxis, :]  # c is shape (1,3)
d = a[:, np.newaxis]  # d is shape (3,1)
e = d.T  # e is shape (1,3)

np.full((3, 1), 1.0)+np.full((1, 3), 1.0)+ 2.0
c@d
d@c
np.concatenate((c, e),)
np.stack((c, e))
np.vstack((c, e))
np.hstack((c, e))

