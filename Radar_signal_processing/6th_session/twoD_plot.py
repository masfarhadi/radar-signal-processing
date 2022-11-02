import numpy as np
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save

np.random.seed(0)
Z=np.random.randn(5,4)
plt.imshow(Z,
           cmap='Oranges',
           vmin=-1,
           origin='lower',
           extent=(-1,1,0,3))
plt.xlabel('Velocity (m/s)')
plt.ylabel('Range (m)')
plt.colorbar(label='Magnitude (dB)')
plt.title('Range/Doppler Map')
plt.plot(0, 1, 'o', label='Mark')
plt.legend(loc='lower center', )#bbox_to_anchor=(0.5, -0.3))
tikz_save('test_imshow.tikz',
          tex_relative_path_to_data='python',
          override_externals=True)
plt.savefig('test_imshow.png', dpi=150)