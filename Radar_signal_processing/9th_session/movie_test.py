import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter as MovieWriter # ffmpeg needed
# make figure as usual
fig=plt.figure()                   # make a new figure (only once!)
np.random.seed(0)                  # reset PRNG to have reproducible a result
lines=plt.plot(np.zeros(4), 'x-')  # dummy data + store handles to modify later
plt.ylim(0, 15)                     # fix ylim to avoid jumpy frames
plt.xticks(np.arange(4))           # manually set ticks on x-axis
plt.grid()
plt.title('Fancy Animation of DC in Noise')
plt.xlabel('Measurement index (1)')
plt.ylabel('Voltage (V)')

moviewriter = MovieWriter(fps=15)            # instanciate moviewriter
with moviewriter.saving(fig, 'test_moviewriter.mp4', dpi=150): # open file
    for j in range(100):                     # loop over frames
        new_data=np.random.rand(4)+9         # generate new data
        lines[0].set_ydata(new_data)         # update plot
        moviewriter.grab_frame()             # append frame
plt.savefig('test_moviewriter.png', dpi=150) # after loop, thus no dummy data