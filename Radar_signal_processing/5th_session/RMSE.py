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
psi = 0.21  # 0.125  #
phi = 0
s_if = amp * np.exp(2j*np.pi*psi*n+1j*phi)  # amp*np.sin(2*np.pi*psi*n+phi)

N_iter = 1000
SNR_range = np.arange(-20, 41, 2)
psi_err_argmax = np.zeros([np.size(SNR_range),N_iter])
psi_err_polyfit = np.zeros([np.size(SNR_range),N_iter])


for n_fft in 2**np.arange(5, 12):
    for SNR_cnt in range(len(SNR_range)):
        SNR = SNR_range[SNR_cnt]
        for iter_cnt in range(N_iter):
            variance = (amp**2)/(2*10**(SNR/10))
            noise = np.sqrt(variance)*np.random.randn(N)
            x_if = s_if + noise

            x_fft = 1/N*np.fft.fft(x_if, n_fft)
            freq = np.linspace(0, fs, int(n_fft), endpoint=False)/(fs)
            fc = np.argmax(np.abs(x_fft))
            pc = np.abs(x_fft[fc ])
            pl = np.abs(x_fft[(fc-1) % n_fft])
            pr = np.abs(x_fft[(fc+1) % n_fft])
            spectrum_excert = np.array([pl, pc, pr])
            df = np.mean(np.diff(freq))
            psi_offset, f_max, polynom_params = calc_polyfit_correction(spectrum_excert, df)
            poly_func = np.poly1d(np.polyfit(freq[fc-1:fc+2], spectrum_excert, 2))
            fq = np.linspace(freq[fc-1], freq[fc+1], n_fft)

            psi_err_argmax[SNR_cnt, iter_cnt] = np.abs(psi - freq[fc])
            psi_err_polyfit[SNR_cnt, iter_cnt] = np.abs(psi - (freq[fc]+psi_offset))

    psi_avg_argmax = np.sum(psi_err_argmax, axis=1)/N_iter
    psi_avg_polyfit = np.sum(psi_err_polyfit, axis=1) / N_iter
    plt.figure('argmax')
    plt.plot(SNR_range,psi_avg_argmax, label='Z=%d'%n_fft,)

    plt.figure('polyfit')
    plt.plot(SNR_range,psi_avg_polyfit, label='Z=%d'%n_fft,)


plt.figure('argmax')
plt.title('argmax $\psi_{0}= %.2f ' %psi)
plt.xlabel('SNR (dB)')
plt.yscale('symlog')
plt.ylabel('RMSE (1)')
#plt.ylim(0.7, 1.1)
#plt.xlim(0.15, 0.22)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


plt.figure('polyfit')
plt.title('polyfit $\psi_{0}$= %.2f ' %psi)
plt.xlabel('SNR (dB)')
plt.yscale('symlog')
plt.ylabel('RMSE (1)')
#plt.ylim(0.7, 1.1)
#plt.xlim(0.15, 0.22)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()