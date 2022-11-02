import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='FFBP measurement data movie')
writer = FFMpegWriter(fps=10, metadata=metadata)

project_path = 'D:\\Project\\rbjkusar'
min_plot_level = -70
fig = plt.figure('results')   # make a new figure (only once!)
#fig = plt.figure()
ffbp_data = np.load(project_path + '\\Figures\\Result figures\\movie_files\\' + str(277) + '_ffbp_data.npy')
img_ffbp = np.abs(20 * np.log10(ffbp_data))  # dummy data + store handles to modify later
img_ffbp = img_ffbp - np.max(img_ffbp)
img_ffbp[img_ffbp < min_plot_level] = min_plot_level
result_img = plt.imshow(img_ffbp, origin='lower', cmap='jet', )

l, = plt.plot([], [], 'k-o')

# plt.xlim(-5, 5)
# plt.ylim(-5, 5)
# x0, y0 = 0, 0

with writer.saving(fig, "writer_test.mp4", dpi=150):
    for i in range(420):
        # x0 += 0.1 * np.random.randn()
        # y0 += 0.1 * np.random.randn()
        ffbp_data = np.load(project_path + '\\Figures\\Result figures\\movie_files\\' + str(i) + '_ffbp_data.npy')
        img_ffbp = np.abs(20 * np.log10(ffbp_data))  # dummy data + store handles to modify later
        img_ffbp = img_ffbp - np.max(img_ffbp)
        img_ffbp[img_ffbp < min_plot_level] = min_plot_level
        result_img.set_data(img_ffbp)
        writer.grab_frame()