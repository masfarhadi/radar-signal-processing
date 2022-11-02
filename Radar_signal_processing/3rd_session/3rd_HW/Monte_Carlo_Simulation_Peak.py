# -*- coding: utf-8 -*-
"""
Radar signal processing homework
Cross correlation
page 14/14 of RaSigUE_slides_3

@author: Mas.farhadi
"""

import numpy as np
import scipy.special
# import scipy.stats
import matplotlib.pyplot as plt


def rect_func(x_sample, max_value):
    """
    create a rect function between -max_value and max_value
    :param x_sample:
    :param max_value:
    :return: rect sequence
    """
    return np.where(abs(x_sample) <= max_value, 1, 0)


def monte_carlo_sim_func(test_signal, sigma_values):
    # %% ---- Monte carlo simulation function  ----
    receive_signal = np.concatenate([np.zeros(10), test_signal, np.zeros(10)])
    n_smaples = len(receive_signal)
    seeds_numbers = 1000
    real_peak_idx = 15
    rmse = np.zeros(len(sigma_values))
    for cnt, sigma in enumerate(sigma_values):
        square_error = 0
        for noise_seed in range(seeds_numbers):
            np.random.seed(noise_seed)  # any integer number will do
            noise_signal = np.sqrt(sigma) * np.random.randn(n_smaples)
            noisy_receive_signal = noise_signal + receive_signal
            noisy_corr_signal = np.correlate(noisy_receive_signal, test_signal, mode='same')
            estimated_peak_idx = np.argmax(noisy_corr_signal)
            square_error = square_error + (estimated_peak_idx - real_peak_idx)**2

        rmse[cnt] = square_error/seeds_numbers

    return rmse


# %% ---- Monte carlo simulation to detect peak in Noise ----
x_seq = np.arange(-5, 6)
test_signal_rect = rect_func(x_seq, 5)
test_signal_barker = np.array([+1, +1, +1, -1, -1, -1, +1, -1, -1, +1, -1])  # barker_11
sigma_values = np.logspace(-1, 1, 21)

rmse_rect = monte_carlo_sim_func(test_signal_rect, sigma_values)
rmse_barker = monte_carlo_sim_func(test_signal_barker, sigma_values)

plt.figure('monte_carlo_simulation')
plt.title('Monte-Carlo Simulation: Peak Localization')
rmse_rect_plot, = plt.semilogx(sigma_values, rmse_rect, '-*', label='RMSE_rect',)
rmse_barker_plot, = plt.semilogx(sigma_values, rmse_barker, '-+', label='RMSE_barker',)

plt.legend(handles=[rmse_rect_plot, rmse_barker_plot])
plt.ylabel('RMSE values')
plt.grid(True)
plt.show()










