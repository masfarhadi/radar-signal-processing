# -*- coding: utf-8 -*-
"""
Radar signal processing homework
Pseudo spectrum
RaSigUE_slides_8

@author: Mas.farhadi
"""

import numpy as np
import scipy.special
import scipy.signal as sp
# import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import axes
from matplotlib2tikz import save as tikz_save
import spectrum

# matplotlib.rcParams['backend'] = 'Qt5Agg'
# %matplotlib qt
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
NA = 8  # number of channels

# Two close (spurious) targets and three wanted targets:
A0_arr = (np.array([20, 18, 0.1, 2, 2, 0.1])*1e-6).reshape(-1, 1)  # magnitudes
r0_arr = np.array([0.001, 0.1, 15, 15, 40, 80]) .reshape(-1, 1)  # ranges
# v0_arr = np.array([0, 0, 1, 1, 0, 28]).reshape(-1, 1)  # range-rates
theta0_arr = (np.array([-3, 5, -35, -31, 0, 10])*np.pi/180.0).reshape(-1, 1)  # angles (degree)

N = int(T/Ts)
n = np.arange(N)  # np.linspace(0, T-Ts, int(T/Ts))

Te = T0*(10**(F_dB/10) - 1)*20
p_n = k_b*Te/T
v_n = np.sqrt(p_n*R)
w_noise = v_n*np.random.randn(NA, N)

# phi_m = 2*fc*r0_arr/c0 + n*Ts*(2*k_rampe*r0_arr + 2*fc*v0_arr)/c0 + ((n*Ts)**2)*(2*k_rampe*v0_arr)/c0
# s_if = np.sum(A0_arr * np.cos(2*np.pi*phi_m), axis=0) + w_noise
# plt.figure('single_chirp')
# plt.title('single_chirp')
# plt.plot(n/1000, s_if, label='$S_{IF}$', )
# #plt.plot(fs*psi/1e6, 20*np.log10(np.abs(amp)), 'o', markersize=6, label='parameter',)
# plt.xlabel('time (ms)')
# plt.ylabel('amplitude (v)')
# plt.legend()
# plt.grid()

s_if_ula = np.zeros((NA, N))
for ant_cnt in range(NA):
    phi_m_ant = 2*fc*r0_arr/c0 + n*Ts*(2*k_rampe*r0_arr)/c0 + 1/2*np.sin(theta0_arr)*ant_cnt
    s_if_ula[ant_cnt, :] = np.sum(A0_arr * np.cos(2 * np.pi * phi_m_ant), axis=0)

x_if_ula = s_if_ula + w_noise
range_window = sp.get_window('hanning', N)
az_window = sp.get_window('boxcar', NA).reshape(-1, 1)
x_if_ula_win = az_window * range_window * x_if_ula

n_fft = 2**10
X_FFT = 1/N * np.fft.fft(x_if_ula_win, n_fft)
range_vec = np.linspace(0, n_fft, n_fft, endpoint=False)*range_bin*N/n_fft

range_limit = 90  # m
X_FFT = X_FFT[:, :int(range_limit/(range_bin*N/n_fft))]
range_vec = range_vec[:int(range_limit/(range_bin*N/n_fft))]

# plot range profile
plt.figure('ULA_FFT spectrum')
plt.plot(range_vec, 20*np.log10(np.abs(X_FFT.T)))
# plt.xlim([0, range_limit])
plt.ylim([-180, -80])
plt.grid()
tikz_save('ULA_FFT spectrum.tikz')
plt.savefig('ULA_FFT spectrum.png', dpi=150)

n_fft_ant = 256  # np.arange(NA)
u_axes = np.linspace(-0.5, 0.5, n_fft_ant)
angle_axes = np.arcsin((2*np.linspace(-n_fft_ant/2, n_fft_ant/2-1, num=n_fft_ant)/n_fft_ant))/np.pi*180
range_axes = range_vec
X_FFT_2D = np.fft.fftshift(1/NA * np.fft.fft(X_FFT, n=n_fft_ant, axis=0), axes=0)

# plot 2D FFT spectrum
plt.figure('ULA FFT 2D')
plt.title('ULA FFT 2D')
plt.imshow(20*np.log10(np.abs(X_FFT_2D.T)), origin='lower', aspect='auto', extent=[u_axes.min(), u_axes.max(), range_axes.min(), range_axes.max()],
           cmap='Oranges', vmin=-180, vmax=-100, )
plt.xlabel('angle  (radian)')
plt.ylabel('Range (m)')
plt.colorbar(label='Magnitude (dB)')
plt.plot(0.5*np.sin(theta0_arr), r0_arr, 'xk')
tikz_save('ULA FFT 2D.tikz')
plt.savefig('ULA FFT 2D.png', dpi=150)

# plot pcolormesh
plt.figure('pcolormesh range/cross-range')
plt.clf()
u_mesh, range_mesh = np.meshgrid(u_axes, range_axes)
theta_mesh = np.arcsin(2*u_mesh)
y_mesh = range_mesh*np.cos(theta_mesh)
x_mesh = range_mesh*np.sin(theta_mesh)
plt.pcolormesh(x_mesh, y_mesh, 20*np.log10(np.abs(X_FFT_2D.T)), vmin=-180, vmax=-100,)

x_marker = r0_arr*np.sin(theta0_arr)
y_marker = r0_arr*np.cos(theta0_arr)
plt.plot(x_marker, y_marker, 'rx')
plt.colorbar(label='Magnitude (dB)')
plt.xlabel('Cross-range (m)')
plt.ylabel('Range (m)')
tikz_save('pcolormesh range cross-range.tikz')
plt.savefig('pcolormesh range cross-range.png', dpi=150)

Tdb = -160  # dbv
X_2D_db = 20*np.log10(np.abs(X_FFT_2D.T))
range_indexes = np.where(np.any(X_2D_db > Tdb, axis=1) == True)[0]

for rng_cnt in range(range_indexes.size):
    plt.figure()
    range_array = X_2D_db[range_indexes[rng_cnt], :]
    range_array = range_array - np.max(range_array)
    plt.plot(angle_axes, range_array, label='FFT ')
    plt.ylim([-100, 10])
    plt.grid()

    # music algorithm
    ZA_fine = n_fft_ant  # 256
    for P in [2, 3]:
        music_data = X_FFT.T[range_indexes[rng_cnt],:]
        res = spectrum.music(music_data, P, NFFT=ZA_fine)[0]
        res_db = 20*np.log10(np.abs(res))
        res_norm = res_db - np.max(res_db)
        plt.plot(angle_axes, res_norm, label='music P=%i'%P, )

    plt.legend()
    plt.title('Range of %f m' %(range_indexes[rng_cnt]*(range_bin*N/n_fft)))
    plt.savefig('Range of %f m .png' %(range_indexes[rng_cnt]*(range_bin*N/n_fft)))


