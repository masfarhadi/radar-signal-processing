# -*- coding: utf-8 -*-
"""
Radar signal processing homework
Radar Equation with Simple Objects
page 12/14 of RaSigUE_slides_2

@author: Mas.farhadi
"""

import numpy as np
import scipy.special
# import scipy.stats
import matplotlib.pyplot as plt

c0 = 3e8  # speed of light
Pt_dBm = 1.0        # transmit power
Pt_dB = 10*np.log10(1e-3*np.power(10, Pt_dBm/10))  # dB = dBm - 30
Gt_dB = 15.0        # gain of the TX antenna
Gr_dB = 15.0        # gain of the RX antenna
Pr_min_dBm = -120.  # minimal required power at the receiver
Pr_min_dB = 10*np.log10(1e-3*np.power(10, Pr_min_dBm/10))  # dB = dBm - 30
fc = 77e9  # carrier frequency 77GHz
wavelength = c0 / fc  # wavelength


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


def rcs_sphere_exact(r, fc, iteration=100):
    """
    RCS of sphere with exact equation
    :param r:
    :param fc:
    :return:
    """
    wavelength = c0 / fc
    k = 2 * np.pi / wavelength
    # rcs = 0
    # for n in range(100):
    n = np.arange(1, iteration)
    rcs = (np.pi * r ** 2) * \
          np.abs((1j / (k * r)) *
                 np.sum(((-1) ** n) * (2 * n + 1) *
                        (((k * r * scipy.special.spherical_jn(n - 1, k * r) - n * scipy.special.spherical_jn(n, k * r))
                          / (k * r * scipy.special.hankel1(n - 1, k * r) - n * scipy.special.hankel1(n, k * r)))
                         - (scipy.special.spherical_jn(n, k * r) / scipy.special.hankel1(n, k * r))
                         )
                        )
                 ) ** 2
    return rcs


dim_parameter = np.array([2.0, 1.0, 0.1, 0.01])
target_range = np.arange(1, 30)
pr_corner = np.zeros((len(dim_parameter), len(target_range)))
pr_sphere = np.zeros((len(dim_parameter), len(target_range)))

for n, x in enumerate(dim_parameter):
    rcs_corner = rcs_corner_reflector(x, fc)
    pr_corner[n, :] = Pt_dB + Gt_dB + Gr_dB + 20*np.log10(wavelength) + 10*np.log10(rcs_corner) - 30*np.log10(4*np.pi) \
                      - 40*np.log10(target_range)
    rcs_sphere = rcs_sphere_exact(x, fc, iteration=100)
    pr_sphere[n, :] = Pt_dB + Gt_dB + Gr_dB + 20*np.log10(wavelength) + 10*np.log10(rcs_sphere) - 30*np.log10(4*np.pi) \
                      - 40*np.log10(target_range)

    plt.figure()
    plt.axhline(y=Pr_min_dB, linestyle='dashed', color='r')
    plt.plot(target_range, pr_corner[n, :],
             target_range, pr_sphere[n, :])
    plt.legend(['minimum power','Pr_corner', 'Pr_sphere'])
    plt.title('Radar Equation with a = %.2f ' % (dim_parameter[n]))
    plt.xlabel('range $(meter)$')
    plt.ylabel('Pr (dB)')
    plt.grid(True)
    plt.show()
