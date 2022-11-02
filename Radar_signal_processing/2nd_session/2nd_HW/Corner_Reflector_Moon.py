# -*- coding: utf-8 -*-
"""
Radar signal processing homework
Corner Reflector on the Moon
page 14/14 of RaSigUE_slides_2

@author: Mas.farhadi
"""

import numpy as np
import scipy.special
# import scipy.stats
import matplotlib.pyplot as plt

c0 = 3e8  # speed of light
Gt_dB = 44.15  # dB
Gr_dB = 44.15  # dB
fc = 10*1e9  # 10GHz
snr_db = 10  # dB
T = 290  # K
BN = 1*1e9  # noise bandwidth GHz
KB = 1.38*1e-23  # boltzmann constant  J/K
a = 0.5  # corner reflector size

def rcs_corner_reflector(a, fc):
    """
    RCS of corner reflector
    :param a:
    :param fc:
    :return: rcs
    """
    wavelength = c0 / fc
    if a > wavelength:
        rcs = 4 * np.pi * a ** 4 / (3 * wavelength ** 2)
    else:
        rcs = np.nan
    return rcs


wavelength = c0 / fc  # wavelength
rcs_corner_moon = rcs_corner_reflector(a, fc)
target_range = 385000*1e3  # 385000Km
pn = 4*KB*T*BN
pn_db = 10*np.log10(pn)
pr_db = snr_db + pn_db


pt_db = pr_db - (Gt_dB + Gr_dB + 20*np.log10(wavelength) + 10*np.log10(rcs_corner_moon) - 30*np.log10(4*np.pi) \
                      - 40*np.log10(target_range))
pt = np.power(10, pt_db/10)
print('require power to detect the signal is %e' %pt)
T_obs = target_range/c0
print('the observation time is %.4f' %T_obs)
