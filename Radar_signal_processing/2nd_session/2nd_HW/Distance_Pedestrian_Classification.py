# -*- coding: utf-8 -*-
"""
Radar signal processing homework
Distance for Pedestrian Classification
page 13/14 of RaSigUE_slides_2

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
dyn_min_dB = Pr_min_dB + 20  # the received power must be 20 dB above the minimum receive power of the system.
fc = 77e9  # carrier frequency 77GHz
wavelength = c0 / fc  # wavelength
rcs_person = 1  # m^2
ped_range = np.arange(1, 30)
pr_person = Pt_dB + Gt_dB + Gr_dB + 20 * np.log10(wavelength) + 10 * np.log10(rcs_person) - 30 * np.log10(4 * np.pi) \
            - 40 * np.log10(ped_range)

plt.figure()
plt.axhline(y=dyn_min_dB, linestyle='dashed', color='r')
plt.plot(ped_range, pr_person)
plt.legend(['minimum power+20dB', 'Pr_person'])
plt.title('Distance for Pedestrian Classification')
plt.xlabel('range $(meter)$')
plt.ylabel('Pr (dB)')
plt.grid(True)
plt.show()
