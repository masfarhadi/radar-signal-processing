# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 18:18:21 2019

@author: farhadi
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib2tikz import save as tikz_save
data = np.reshape(np.arange(6*5), (6, 5))  # dummy data
# map to a circular ring sector
theta = np.linspace(-np.pi, np.pi/2, data.shape[0])
theta = theta[:, np.newaxis]            # make theta a column vector
r = np.linspace(4, 12, data.shape[1])
X=r*np.sin(theta)
Y=r*np.cos(theta)  # list x column vector produces a matrix plot
c=plt.pcolormesh(X, Y, data)         # plot it
plt.colorbar()                       # add colorbar
plt.plot(X, Y, 'kx')                 # combination with other plots possible
plt.grid()                           # common plot formatting stuff works too
# export
tikz_save('test_pcolormesh.tikz',
          tex_relative_path_to_data='python',
          override_externals=True)
plt.savefig('test_pcolormesh.png', dpi=150)