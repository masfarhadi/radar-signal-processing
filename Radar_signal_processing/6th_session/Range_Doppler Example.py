# -*- coding: utf-8 -*-
"""
Radar signal processing homework
Range/Doppler example
RaSigUE_slides_6

@author: Mas.farhadi
"""

import numpy as np
import scipy.special
import scipy.signal as sp
# import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import axes
from tikzplotlib import save as tikz_save
import spectrum

plt.rcParams['axes.formatter.limits'] = [-4, 4]

c0 = 3*1e8  # speed of light
k_b = 1.38064852*1e-23  # Boltzmann constant

fs = 10e6  # IF sample frequency
B = 250e6  # RF bandwidth
fc = 24.125e9  # carrier frequency
T = 50e-6  # ramp (chirp) duration
Tp = 100e-6  # chirp repetition rate
Np = 64  # number of pulses
Temp_sys = 9e5  # equivalent noise temperature
R = 50  # reference impedance

Ts = 1/fs  # sampling interval
k_rampe = B/T  # ramp slope
range_bin = c0/(2*B)
n_fft = 2**16

A0_arr = np.array([20, 18, 0.1, 2, 0.1, 4]) * 1e-3  # magnitudes
r0_arr = np.array([0.001, 0.1, 15, 40, 80, 50])  # ranges
v0_arr = np.array([0, 0, 1, 0, 28, -84])  # velocities

n_p = np.arange(Np).reshape(-1, 1)
Ns = int(T/Ts)
n_s = np.arange(Ns)
s_if = np.zeros([Np, Ns])
for m in range(r0_arr.size):
    s_if = s_if + A0_arr[m] * np.cos(2 * np.pi *
                                     (n_p * Tp * 2 * fc * v0_arr[m] / c0
                                      + n_s * Ts * (2 * k_rampe * r0_arr[m] / c0 + 2 * fc * v0_arr[m] / c0)
                                      + 2 * fc * r0_arr[m] / c0 + ((n_s * Ts) ** 2) * 2 * k_rampe * v0_arr[m] / c0
                                      + n_p * n_s * Ts * Tp * 2 * k_rampe * v0_arr[m] / c0))
wind_fast = sp.get_window('nuttall', Ns)
wind_slow = sp.get_window('hanning', Np).reshape(-1, 1)
sif_win = wind_slow * wind_fast * s_if

plt.figure('IF signal')
plt.imshow(sif_win.T, origin='lower', aspect='auto')  # extent=(0, Np, 0, Ns)
plt.xlabel('Range_rate or slow time index (m/s)')
plt.ylabel('Range or fast time index  (m)')
plt.colorbar(label='Magnitude (dB)')
plt.title('Range/Doppler Map')
#plt.plot(0, 1, 'o', label='Mark')
#plt.legend(loc='lower center')
tikz_save('rd_plot.tikz',
          tex_relative_path_to_data='python',
          override_externals=True)
plt.savefig('test_imshow.png', dpi=150)

plt.figure('Range_Doppler_map')
plt.title('Range_Doppler_map')
rd_map = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft2(sif_win.T, s=[n_fft, Np]), axes=1)))
rd_map = rd_map - np.max(rd_map)
x_axes = n_p.T-Np/2
y_axes = np.linspace(0, n_fft-1, n_fft)*range_bin*Ns/n_fft  # y_axes = n_s*range_bin*n_fft/Ns
plt.imshow(rd_map,  origin='lower', aspect='auto', extent=[x_axes.min(), x_axes.max(), y_axes.min(), y_axes.max()],
           cmap='Oranges', vmin=-170, vmax=-30,)
plt.xlabel('Range_rate or slow time index (m/s)')
plt.ylim([0, 100])
plt.ylabel('Range or fast time index  (m)')
plt.colorbar(label='Magnitude (dB)')
#plt.axis([x_axes.min(), x_axes.max(), y_axes.min(), y_axes.max()])
# plt.pcolormesh(x_axes, y_axes, rd_map)#, origin='lower', cmap='Oranges', aspect='auto')

plt.title('Range/Doppler Map')
#plt.plot(0, 1, 'o', label='Mark')
#plt.legend(loc='lower center')
plt.plot(v0_arr[:-1], r0_arr[:-1], 'xb' )
tikz_save('rd_plot.tikz',
          tex_relative_path_to_data='python',
          override_externals=True)
plt.savefig('test_imshow.png', dpi=150)


