import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# generate data
x = np.arange(-15, 15)
y = np.arange(-20, 20)  # generate axis
X, Y = np.meshgrid(x, y)  # generate grid
A = np.cos(2*np.pi*1/x.size*X)*np.cos(2*np.pi*0.25/y.size*Y)**2  # generate data
# init figure
fig = plt.figure(1, figsize=[9, 6])   # default figure size is too small for 3D
fig.clear()                        # reusing same figure with %matplotlib qt
ax = fig.gca(projection='3d')        # set axes to a 3D axis
ax.view_init(elev=27, azim=-66)    # set view
plt.tight_layout(pad=0.0, rect=(-0.1, 0, 1, 1))  # adjust boarders around axes

# plot objects
ax.plot_surface(X, Y, A, alpha=0.3, edgecolor='k')  # surface plot
ax.contour(X, Y, A, zdir='z', offset=-1)  # contour at bottom
ax.contourf(X, Y, A, zdir='x', offset=x[0])  # filled contour side

# plot lines
ax.plot3D(x, np.full(x.shape, np.mean(y)), zs=A[A.shape[0]//2, :], color='r', linewidth=4, label='center')
ax.plot3D(np.full(y.shape,x[0]), y, zs=np.max(A, axis=1), color='k', linewidth=4)
ax.plot3D(np.full(y.shape,x[0]), y, zs=np.min(A, axis=1), linestyle='--', color='k', linewidth=4)

# annotate plot
plt.xlim(x[0], x[-1])
plt.ylim(y[0], y[-1])  # to place contours being in planes
ax.set_zlim(bottom=-1.0, top=1.5*np.max(A))  # zlim is only available in axis
ax.set_zlabel('Magnitude (1)')  # same for zlabel
plt.xlabel('First dimension (1)')
plt.ylabel('Second dimension (2)')
plt.legend()                               # legend not avail. for all items
plt.savefig('test_3D_plot.png', dpi=150)

