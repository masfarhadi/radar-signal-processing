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


def barker_auto_corr_func():
    barker_11 = np.array([+1, +1, +1, -1, -1, -1, +1, -1, -1, +1, -1])
    x_seq = np.arange(-5, 6)
    rect_11 = rect_func(x_seq, max_value=5)
    plt.figure('auto correlation ')
    plt.subplot(222)
    markerline1,_,_ = plt.stem(x_seq, rect_11, label = 'rect_signal',)
    plt.setp(markerline1, 'markerfacecolor', 'b')
    plt.title('rect')
    plt.xlim(-11, 11)
    plt.legend()
    plt.grid(True)

    plt.subplot(221)
    markerline1, _, _ =plt.stem(x_seq, barker_11, label='barker_11',linefmt='-.')
    plt.setp(markerline1, 'markerfacecolor', 'r')
    plt.legend()
    # plt.xlabel('sequence samples')
    plt.xlim(-11, 11)
    plt.ylabel('signal')
    plt.title('Barker11')
    plt.grid(True)

    y_seq = np.arange(-10, 11)
    barker_corr = np.correlate(barker_11, barker_11,'full')
    rect_corr = np.correlate(rect_11, rect_11,'full')
    plt.figure('auto correlation ')
    plt.subplot(224)
    markerline1,_,_ = plt.stem(y_seq, rect_corr, label = 'rect_corr',)
    plt.setp(markerline1, 'markerfacecolor', 'b')
    plt.legend()
    plt.xlim(-11, 11)
    plt.grid(True)
    plt.subplot(223)
    markerline1, _, _ =plt.stem(y_seq, barker_corr, label='barker_corr',linefmt='-.')
    plt.setp(markerline1, 'markerfacecolor', 'r')
    plt.legend()
    # plt.xlabel('sequence samples')
    plt.xlim(-11, 11)
    plt.ylabel('Auto-correlation values')
    plt.grid(True)
    plt.show()

# %% ---- Cross-Correlation to Detect a Signal ----
x_seq = np.arange(-5,6)
test_signal = rect_func(x_seq, 5)
receive_signal = np.concatenate([np.zeros(20), test_signal, np.zeros(39), -test_signal, np.zeros(20)])
plt.figure('Cross_correlation')
plt.subplot(211)
markerline1, _, _ = plt.stem(receive_signal, label='receive_signal', )
plt.setp(markerline1, 'markerfacecolor', 'b')
plt.legend()
# plt.xlabel('samples')
# plt.xlim(-11, 11)
plt.ylabel('signal values')
plt.grid(True)
plt.show()

corr_signal = np.correlate(receive_signal, test_signal, mode='same')
plt.figure('Cross_correlation')
plt.subplot(212)
markerline1, _, _ =plt.stem(corr_signal, label='corr_signal',linefmt='-.')
plt.setp(markerline1, 'markerfacecolor', 'r')
plt.legend()
plt.xlabel('samples')
# plt.xlim(-11, 11)
plt.ylabel('cross-correlation values')
plt.grid(True)
plt.show()


# %% ---- Cross-Correlation to Detect a Signal in Noise ----
sigma = 0.8
n_smaples = len(receive_signal)
np.random.seed(2)  # any integer number will do
noise_signal = np.sqrt(sigma)*np.random.randn(n_smaples)
plt.figure('noise_signal_rect')
plt.hist(noise_signal)#, bins=int(np.sqrt(n_smaples)))
plt.xlabel('Values (1)')
plt.ylabel('Frequency (1)')

noisy_receive_signal = noise_signal + receive_signal
plt.figure('noisy_cross_correlation_rect')
plt.subplot(211)
plt.title('noisy_rect')
markerline1, _, _ = plt.stem(noisy_receive_signal, label='noisy_receive_signal', )
plt.setp(markerline1, 'markerfacecolor', 'b')
plt.legend()
# plt.xlabel('samples')
# plt.xlim(-11, 11)
plt.ylabel('signal values')
plt.grid(True)
plt.show()

noisy_corr_signal = np.correlate(noisy_receive_signal, test_signal, mode='same')
plt.figure('noisy_cross_correlation_rect')
plt.subplot(212)
plt.title('noisy_cross_correlation_rect')
markerline1, _, _ =plt.stem(noisy_corr_signal, label='noisy_corr_signal',linefmt='-.')
plt.setp(markerline1, 'markerfacecolor', 'r')
plt.legend()
plt.xlabel('samples')
# plt.xlim(-11, 11)
plt.ylabel('cross-correlation values')
plt.grid(True)
plt.show()

# %% ---- Cross-Correlation to Detect a Signal in Noise by using the Barker 11 Sequence ----
sigma = 0.8
test_signal_barker = np.array([+1, +1, +1, -1, -1, -1, +1, -1, -1, +1, -1])  # barker_11
receive_signal_barker = np.concatenate([np.zeros(20), test_signal_barker, np.zeros(39), -test_signal_barker, np.zeros(20)])
n_smaples = len(receive_signal_barker)
np.random.seed(1)  # any integer number will do
noise_signal = np.sqrt(sigma)*np.random.randn(n_smaples)
plt.figure('noise_signal_barker')
plt.hist(noise_signal)#, bins=int(np.sqrt(n_smaples)))
plt.xlabel('Values (1)')
plt.ylabel('Frequency (1)')

noisy_receive_signal = noise_signal + receive_signal_barker
plt.figure('noisy_cross_correlation_barker')
plt.subplot(211)
plt.title('noisy_barker')
markerline1, _, _ = plt.stem(noisy_receive_signal, label='noisy_receive_signal', )
plt.setp(markerline1, 'markerfacecolor', 'b')
plt.legend()
# plt.xlabel('samples')
# plt.xlim(-11, 11)
plt.ylabel('signal values')
plt.grid(True)
plt.show()

noisy_corr_signal = np.correlate(noisy_receive_signal, test_signal_barker, mode='same')
plt.figure('noisy_cross_correlation_barker')
plt.subplot(212)
plt.title('noisy_cross_correlation_barker')
markerline1, _, _ =plt.stem(noisy_corr_signal, label='noisy_corr_signal',linefmt='-.')
plt.setp(markerline1, 'markerfacecolor', 'r')
plt.legend()
plt.xlabel('samples')
# plt.xlim(-11, 11)
plt.ylabel('cross-correlation values')
plt.grid(True)
plt.show()