# -*- coding: utf-8 -*-
"""
Radar signal processing homework
Compare RCS of a Sphere and a Corner Reflector
page 11/14 of RaSigUE_slides_2

@author: Mas.farhadi
"""

import numpy as np
import scipy.special
import matplotlib.pyplot as plt

c0 = 3e8  # speed of light


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


def rcs_sphere_approx(r, fc):
    """
    RCS approximation of sphere
    :param r:
    :param fc:
    :return:
    """
    wavelength = c0 / fc
    if (2 * np.pi * r) >= wavelength:
        rcs = np.pi * r ** 2
    else:
        k = 2 * np.pi / wavelength
        rcs = 9 * ((k * r) ** 4) * np.pi * r ** 2

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


def rcs_calc_func():
    fc = 1e9  # center frequency 1GHz
    wavelength = c0 / fc  # wavelength

    dim_parameter = np.logspace(-2, 1, num=100) * wavelength
    rcs_corner = np.zeros(len(dim_parameter))
    rcs_sphere = np.zeros(len(dim_parameter))
    rcs_sphere_aprx = np.zeros(len(dim_parameter))

    for n, x in enumerate(dim_parameter):
        rcs_corner[n] = rcs_corner_reflector(x, fc)
        rcs_sphere[n] = rcs_sphere_exact(x, fc, iteration=100)
        rcs_sphere_aprx[n] = rcs_sphere_approx(x, fc)

    plt.figure('Compare_RCS')
    plt.loglog(dim_parameter, rcs_corner,
               dim_parameter, rcs_sphere,
               dim_parameter, rcs_sphere_aprx)
    plt.legend(['RCS_corner', 'RCS_sphere', 'RCS_sphere_aprx'])
    plt.title('Radar Cross Section')
    plt.xlabel('size of object $(\\times \lambda)$')
    plt.ylabel('RCS (dBsm)')
    plt.grid(True)
    plt.show()
    plt.savefig('2nd_HW_Compare_RCS.png')


rcs_calc_func()


