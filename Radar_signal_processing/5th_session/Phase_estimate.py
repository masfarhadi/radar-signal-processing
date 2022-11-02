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
psi = 0.14  # 0.125  # 0.19  #
phi = 0.85
x = amp * np.exp(2j*np.pi*psi*n+1j*phi)  # amp*np.sin(2*np.pi*psi*n+phi)


for n_fft in 2**np.arange(5, 12):
    x_fft = 1/N*np.fft.fft(x, n_fft)
    freq = np.linspace(0, fs, int(n_fft), endpoint=False)/(fs)
    fc = np.argmax(np.abs(x_fft))
    pc = np.abs(x_fft[fc])
    pl = np.abs(x_fft[fc-1])
    pr = np.abs(x_fft[fc+1])
    spectrum_excert = np.array([pl, pc, pr])
    df = np.mean(np.diff(freq))
    psi_offset, f_max, polynom_params = calc_polyfit_correction(spectrum_excert, df)
    poly_func = np.poly1d(np.polyfit(freq[fc-1:fc+2], spectrum_excert, 2))
    fq = np.linspace(freq[fc-1], freq[fc+1], n_fft)

    # calculate the  DFT sum for peak frequency of poly fit
    dft_sum = 1 / N * np.sum(x * np.exp(-2j * np.pi * (freq[fc]+psi_offset) * n))
    phase_poly_max = np.angle(dft_sum)

    phase_argmax = np.angle(x_fft[fc])

    phi_err_argmax = np.abs(phi - phase_argmax)
    phi_err_polyfit = np.abs(phi - phase_poly_max)

    plt.figure()
    plt.title('N= %d, Z= %d, $\phi_{err,argmax} / \phi_{err,polyfit}$: %.2e/%.2e ' %(N, n_fft, phi_err_argmax, phi_err_polyfit))
    plt.plot(freq[fc-3:fc+3], np.angle(x_fft[fc-3:fc+3]), '-x', label='FFT_spectrum',)
    plt.plot(psi, phi, 'o', markersize=6, label='True_parameter',)
    # plt.plot(fq, poly_func(fq), '-.', label='Polynomial')
    plt.plot(freq[fc] + psi_offset, phase_poly_max, 'x', markersize=6, label='Poly_max', )
    plt.plot(freq[fc], phase_argmax, '+', markersize=6, label='Arg max',)
    plt.xlabel('Normalized Frequency (1)')
    plt.ylabel('Phase (rad)')
    #plt.ylim(0.7, 1.1)
    #plt.xlim(0.15, 0.22)
    plt.legend()
    plt.grid()
    plt.show()



    # freqs, results = goertzel(x, 1,(0.85))
    # freqs_grtz = np.asarray(freqs)
    # results_grtz = np.asarray(results)
    #
    # real_grtz = results_grtz[:,0]
    # imag_grtz = results_grtz[:,1]
    # spect_grtz = results_grtz[:,2]
    #
    # fc_grtz = np.argmax(spect_grtz)
    # phase_grtz = np.angle(real_grtz+1j*imag_grtz)

    # plt.figure()
    # plt.plot(freqs_grtz,20*np.log10(spect_grtz/np.max(spect_grtz)))

#
# # goertzel function is taken from : https://gist.github.com/sebpiq/4128537
# import math
# def goertzel(samples, sample_rate, *freqs):
#     """
#     Implementation of the Goertzel algorithm, useful for calculating individual
#     terms of a discrete Fourier transform.
#     `samples` is a windowed one-dimensional signal originally sampled at `sample_rate`.
#     The function returns 2 arrays, one containing the actual frequencies calculated,
#     the second the coefficients `(real part, imag part, power)` for each of those frequencies.
#     For simple spectral analysis, the power is usually enough.
#     Example of usage :
#
#         freqs, results = goertzel(some_samples, 44100, (400, 500), (1000, 1100))
#     """
#     window_size = len(samples)
#     f_step = sample_rate / float(window_size)
#     f_step_normalized = 1.0 / window_size
#
#     # Calculate all the DFT bins we have to compute to include frequencies
#     # in `freqs`.
#     bins = set()
#     for f_range in freqs:
#         f_start, f_end = f_range
#         k_start = int(math.floor(f_start / f_step))
#         k_end = int(math.ceil(f_end / f_step))
#
#         if k_end > window_size - 1: raise ValueError('frequency out of range %s' % k_end)
#         bins = bins.union(range(k_start, k_end))
#
#     # For all the bins, calculate the DFT term
#     n_range = range(0, window_size)
#     freqs = []
#     results = []
#     for k in bins:
#
#         # Bin frequency and coefficients for the computation
#         f = k * f_step_normalized
#         w_real = 2.0 * math.cos(2.0 * math.pi * f)
#         w_imag = math.sin(2.0 * math.pi * f)
#
#         # Doing the calculation on the whole sample
#         d1, d2 = 0.0, 0.0
#         for n in n_range:
#             y = samples[n] + w_real * d1 - d2
#             d2, d1 = d1, y
#
#         # Storing results `(real part, imag part, power)`
#         results.append((
#             0.5 * w_real * d1 - d2, w_imag * d1,
#             d2 ** 2 + d1 ** 2 - w_real * d1 * d2)
#         )
#         freqs.append(f * sample_rate)
#     return freqs, results