# -*- coding: utf-8 -*-
"""
Radar signal processing homework
The Fast Fourier Transform
RaSigUE_slides_5

@author: Mas.farhadi
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp


def calc_polyfit_correction(spectrum_excert, df):
     """
     Parameters
     ----------
     spectrum_excert: list of three floats
     left, center and right magnitude value
     df: float
     frequency difference of two consecutive FFT-bins
     Returns
     -------
     offset_frequency: float
     Offset towards the location of Fc
     Fmax: magnitude at the maximum
     a: list of three floats
     Parameters of the polynomial.
     """
     pl = spectrum_excert[0]
     pc = spectrum_excert[1]
     pr = spectrum_excert[2]
     a0 = pc
     a1 = (pr - pl)/(2*df)
     a2 = (pl - 2*pc + pr)/(2*df**2)
     psi_offset = -a1/(2*a2)  # -df*(pr-pl)/(2*(pl-2*pc+pl))  #
     f_max = (-2*pr*(4*pc+pl)+(pl-4*pc)**2+pr**2)/(16*pc-8*(pl+pr))

     return psi_offset, f_max, [a2, a1, a0]


# %% ---- Test Polynomial Fit ----
N = 32
n = np.arange(N)
fs = 5*1e6  # sampling frequency is 5 MHz
Ts = 1 / fs
amp = 1
psi = 0.19  # 0.125  #
phi = 0.85
x = amp * np.exp(2j*np.pi*psi*n+1j*phi)  # amp*np.sin(2*np.pi*psi*n+phi)

for n_fft in 2**np.arange(5, 12):
    x_fft = 1/N*np.fft.fft(x, n_fft)
    freq = np.linspace(0, fs, int(n_fft), endpoint=False)/(fs)
    fc = np.argmax(np.abs(x_fft))# [:int(n_fft / 2)]
    pc = np.abs(x_fft[fc])
    pl = np.abs(x_fft[fc-1])
    pr = np.abs(x_fft[fc+1])
    spectrum_excert = np.array([pl, pc, pr])
    df = np.mean(np.diff(freq))
    psi_offset, f_max, polynom_params = calc_polyfit_correction(spectrum_excert, df)
    poly_func = np.poly1d(np.polyfit(freq[fc-1:fc+2], spectrum_excert, 2))
    fq = np.linspace(freq[fc-1], freq[fc+1], n_fft)

    psi_err_argmax = np.abs(psi - freq[fc])
    psi_err_polyfit = np.abs(psi - (freq[fc]+psi_offset))

    plt.figure()
    plt.title('N= %d, Z= %d, $\psi_{err,argmax} / \psi_{err,polyfit}$: %.2e/%.2e ' %(N, n_fft, psi_err_argmax, psi_err_polyfit))
    plt.plot(freq[fc-3:fc+3], np.abs(x_fft[fc-3:fc+3]), '-', label='FFT_spectrum',)
    plt.plot(psi, np.abs(amp), 'o', markersize=6, label='True_parameter',)
    plt.plot(fq, poly_func(fq), '-.', label='Polynomial')
    plt.plot(freq[fc] + psi_offset,  f_max, 'x', markersize=6, label='Poly_max',)
    plt.plot(freq[fc], np.abs(pc), '+', markersize=6, label='Arg max',)
    plt.xlabel('Normalized Frequency (1)')
    plt.ylabel('Power (dB)')
    #plt.ylim(0.7, 1.1)
    #plt.xlim(0.15, 0.22)
    plt.legend()
    plt.grid()
    plt.show()





# ######################### Main #########################
# A = np.array([1])
# psi = np.array([0.19])
# phi = np.array([0.85])
# N = 32
# Z = 4*N
# t = np.arange(N)
# x_sig = A * np.sin(2*np.pi*psi*t + phi)
# x_fft = np.fft.fft(x_sig)
# #df =
# xfft2 = calc_polyfit_correction(x_fft, df)

    # plot fft spectrum
    # plt.figure('Visualization_FFT_Spectrum')
    # plt.title('Visualization_FFT_Spectrum')
    # plt.plot(freq, 20 * np.log10(1 / n_fft * np.abs(x_fft[:int(n_fft / 2)])), '-x', label='FFT_spectrum', )
    # plt.plot(fs*psi/1e6, 20*np.log10(np.abs(amp)), 'o', markersize=6, label='True_parameter',)
    # plt.xlabel('Frequency (MHz)')
    # plt.ylabel('Power (dB)')
    # plt.legend()
    # plt.grid()
    # plt.show()



# # %% ---- Calculate Polynomial Fit ----
#
#
# def calc_polyfit_correction(spectrum_exert, df):
#     """
#     Parameters
#     ----------
#     spectrum_exert: list of three floats
#         left, center and right magnitude value
#     df: float
#         frequency difference of two consecutive FFT-bins
#     Returns
#     -------
#     offset_frequency: float
#         Offset towards the location of Fc
#     Fmax: magnitude at the maximum
#     a: list of three floats
#         Parameters of the polynomial.
#     """
#     p_l = spectrum_exert[0]
#     p_c = spectrum_exert[1]
#     p_r = spectrum_exert[2]
#
#     psi_offset = df*(p_r - p_l)/(2*(p_l - 2*p_c + p_r))
#     f_max = (-2*p_r*(4*p_c + p_l) + (p_l - 4*p_c)**2 + p_r**2)/(16*p_c - 8*(p_l+p_r))
#     a0 = p_c
#     a1 = (p_r - p_l)/(2*df)
#     a2 = (p_l - 2*p_c + p_r)/(2*df**2)
#     return psi_offset, f_max, [a2, a1, a0]