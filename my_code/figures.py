# figure of personal milestones

import matplotlib.pyplot as plt

epochs = [162, 152, 151, 149, 136, 122, 120, 93, 91, 83, 80, 55, 40, 21, 19, 18, 16, 14, 13]
kappas = [0.733, 0.72, 0.707, 0.68, 0.669, 0.653, 0.649, 0.62, 0.555, 0.579, 0.532, 0.487, 0.45, 0.42, 0.41, 0.354, 0.33, 0.28, 0]
plt.plot(epochs, kappas, linestyle='--', marker='o', color='b')
plt.ylabel("Max Kappa")
plt.xlabel("Experiment Number")

reasons = ['256px', '192px + extra ConvPool', '192px + extra Pool', '152px', 'kappa weighted error func', '+/-30 color cast', '+/-20 color cast', 'flips', '4 outputs', 'color', 'nnrank-re', 'controlled batch distributions', 'LReLu', 'Overlap Pooling', '1 more FC dropout', 'Both FC pooling', 'All Conv dropout', 'GlorotUniform Init', 'vgg_mini7']
# import pylab
# pylab.xticks(epochs, reasons, rotation=60, fontsize=8)

plt.show()

# figure of pathological cases

from mpl_toolkits.axes_grid1 import AxesGrid
from skimage.io import imread

fig = plt.figure()
grid = AxesGrid(fig, 111, nrows_ncols = (1, 4))
names = ['23050_right.png', '2468_left.png', '15450_left.png', '406_left.png']
imgs = [imread(n) for n in names]
[grid[i].imshow(imgs[i]) for i in range(len(imgs))]
plt.axis('off')
plt.savefig('out.png', dpi=300)

