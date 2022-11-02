# -*- coding: utf-8 -*-
"""
Radar signal processing homework
The FMCW single chirp with ULA
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
from matplotlib.animation import FFMpegWriter as MovieWriter # ffmpeg needed
from matplotlib2tikz import save as tikz_save
import spectrum

# matplotlib.rcParams['backend'] = 'Qt5Agg'
# %matplotlib qt
plt.rcParams['axes.formatter.limits'] = [-4, 4]

def get_target_params(t):
    """
    Calculates targets parameters for a given point in time.
    Parameters
    ----------
    t : float
    Time in seconds for which the target parameters should be calculated.
    Returns
    -------
    Tuple containing target parameters as numpy arrays.
    A0_arr : 0D-array
    holding magnitudes in V
    r0_arr : 0D-array
    holding ranges in m
    theta0_arr : 0D-array
    holding angles of incident in rad
    """
    # add static targets
    r0_lst = [0.001, 0.1, 15]
    A0_lst_uV = [20, 18, 0.1]
    theta0_lst_deg = [-3, 5, -35]
    # add a target moving radially
    r0_lst.append(12 + 1 * t)
    A0_lst_uV.append(15 ** 4 / r0_lst[-1] ** 4)
    theta0_lst_deg.append(31)
    # add a target moving in a circle
    r0_lst.append(40)
    A0_lst_uV.append(2)
    theta0_lst_deg.append(-25 + 2 * t)
    # convert lists to arrays and return them
    A0_arr = (np.array(A0_lst_uV) * 1e-6).reshape(-1, 1)
    r0_arr = np.array(r0_lst).reshape(-1, 1)
    theta0_arr = (np.array(theta0_lst_deg) * np.pi / 180).reshape(-1, 1)
    return A0_arr, r0_arr, theta0_arr


# constant values
c0 = 3*1e8  # speed of light
k_b = 1.38064852*1e-23  # Boltzmann constant

system_parameters = {
    # system parameters for a single chirp FMCW system
    'fs': 1e6,  # IF sample frequency
    'B': 250e6,  # RF bandwidth
    'fc': 24.125e9,  # carrier frequency
    'T': 1e-3,  # chirp duration
    'NF_dB': 12,  # equivalent noise temperature
    'T0': 290,  # system temperature
    'R': 50,  # reference impedance
    'NA': 8,  # number of ULA channels
    'ZA': 64,  # zero-padding to ZA FFT bins in DoA
    'T_dB': -160,  # threshold in dBV
    }

Ts = 1/system_parameters['fs']  # 1/fs  # sampling interval
k_rampe = system_parameters['B']/system_parameters['T']  # ramp slope
F_dB = 1      # noise figure of the system  # 12
range_bin = c0/(2*system_parameters['B'])
N = int(system_parameters['T']/Ts)
n = np.arange(N)  # np.linspace(0, T-Ts, int(T/Ts))

# observation timestamps, i.e. times where chirps are sent
t_arr = np.linspace(0, 35, 351)

Te = system_parameters['T0']*(10**(F_dB/10) - 1)*20
p_n = k_b*Te/system_parameters['T']
v_n = np.sqrt(p_n*system_parameters['R'])
w_noise = v_n*np.random.randn(system_parameters['NA'], N)

s_if_ula = np.zeros((system_parameters['NA'], N))

n_fft = N  # 2**10
range_window = sp.get_window('hanning', N)
az_window = sp.get_window('boxcar', system_parameters['NA']).reshape(-1, 1)

range_vec = np.linspace(0, n_fft, n_fft, endpoint=False)*range_bin*N/n_fft
range_limit = 50  # m
range_vec = range_vec[:int(range_limit/(range_bin*N/n_fft))]

n_fft_ant = system_parameters['ZA']
u_axes = np.linspace(-0.5, 0.5, n_fft_ant, endpoint=False)  # angle_axes = np.arcsin((2*np.linspace(-n_fft_ant/2,n_fft_ant/2-1, num=n_fft_ant)/n_fft_ant))/np.pi*180
range_axes = range_vec

A0_arr, r0_arr, theta0_arr = get_target_params(0)
# plot range profile
X_FFT = np.ones([n_fft, range_vec.size])
fig = plt.figure('ULA_FFT spectrum')
plt.clf()
Lines = plt.plot(range_vec, 20*np.log10(np.abs(X_FFT.T)))
Marker = plt.plot(r0_arr, 20*np.log10(np.abs(A0_arr)), 'kx')
plt.xlim([0, range_limit])
plt.ylim([-180, -80])
plt.grid(True)

moviewriter = MovieWriter(fps=15)            # instanciate moviewriter
with moviewriter.saving(fig, 'Target_simulator_moviewriter.mp4', dpi=150):  # open file
    for time_cnt in range(t_arr.size):
        A0_arr, r0_arr, theta0_arr = get_target_params(t_arr[time_cnt])

        for ant_cnt in range(system_parameters['NA']):
            phi_m_ant = 2*system_parameters['fc']*r0_arr/c0 + n*Ts*(2*k_rampe*r0_arr)/c0 + 1/2*np.sin(theta0_arr)*ant_cnt
            s_if_ula[ant_cnt, :] = np.sum(A0_arr * np.cos(2 * np.pi * phi_m_ant), axis=0)

        x_if_ula = s_if_ula + w_noise
        x_if_ula_win = az_window * range_window * x_if_ula

        X_FFT = 1/N * np.fft.fft(x_if_ula_win, n_fft)
        X_FFT = X_FFT[:, :int(range_limit/(range_bin*N/n_fft))]

        for ant_cnt in range(system_parameters['NA']):
            Lines[ant_cnt].set_ydata(20*np.log10(np.abs(X_FFT[ant_cnt, :])))  # update plot

        Marker[0].set_data(r0_arr, 20*np.log10(np.abs(A0_arr)))

        moviewriter.grab_frame()  # append frame

# tikz_save('ULA_FFT spectrum.tikz')
plt.savefig('Target simulator.png', dpi=150)