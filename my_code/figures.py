# -*- coding: utf-8 -*-

# This file makes figures for about.md from data saved locally on my machine

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from skimage.io import imread
from os import listdir, path
import re
import cPickle
import numpy

import pdb

# figure of personal milestones

def plot_personal_milestones():
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

def plot_pathological_imgs():
    fig = plt.figure()
    grid = AxesGrid(fig, 111, nrows_ncols = (1, 4))
    names = ['23050_right.png', '2468_left.png', '15450_left.png', '406_left.png']
    imgs = [imread(n) for n in names]
    [grid[i].imshow(imgs[i]) for i in range(len(imgs))]
    plt.axis('off')
    plt.savefig('out.png', dpi=300)

# figure of all kappa curves

# skips 1 lost result from above, is in diff order
def plot_validation_summary(mode='kappa'):
    labels = ['GlorotUniform Init', 'All Conv dropout', 'Both FC pooling', '1 more FC dropout', 'Overlap Pooling', 'LReLu', 'controlled batch distributions', 'nnrank-re', 'color', '4 outputs', 'random flips', '+/-20 color cast', '+/-30 color cast', 'kappa weighted error func', '152px', '192px + extra Pool', '192px + extra ConvPool', '256px']
    result_file_ids = ['9810402b', '43c1ce27', '3ad18dc4', 'd0ceae40', '37f51128', '4f03224a', '9a13f109', '0e826152', 'd5d6d371', '7a316e9c', 'f46c4cc3', 'ea52abf4', '36c79ff6', 'd4f01372', '4cab8f47', 'c2eee209', '5c27cce7', '43184304']
    result_path = 'results/'

    plot_idx_ranges = [range(0,6), range(5,12), range(11,18)]

    for plot_idx_range in plot_idx_ranges:
        plt.clf()
        last_idx = list(plot_idx_range)[-1]
        for i in plot_idx_range:
            id_ = result_file_ids[i]
            filenames = [f for f in listdir(result_path) if re.search(('%s.*pkl' % id_), f)]
            if not (len(filenames) == 1):
                raise ValueError("%s's files': {}".format(filenames) % id_)
            f = open(result_path+filenames[0])
            historical_train_losses, historical_val_losses, historical_val_kappas, n_iter_per_epoch = cPickle.load(f)
            n_iter_per_epoch = float(n_iter_per_epoch)

            train = numpy.array(historical_train_losses)
            valid = numpy.array(historical_val_losses)
            kappa = numpy.array(historical_val_kappas)
            # scale iter to epoch
            train[:,0] = train[:,0] / n_iter_per_epoch
            valid[:,0] = valid[:,0] / n_iter_per_epoch
            kappa[:,0] = kappa[:,0] / n_iter_per_epoch

            label = labels[i]
            if mode == 'kappa':
                plt.plot(kappa[:,0], kappa[:,1], label=label)
                plt.ylabel("Kappa on Validation")
            else:
                plt.plot(kappa[:,0], valid[:,1] / valid[0,1], label=label)
                plt.ylabel("Standardized Error on Validation")

            if i == last_idx:
                plt.xlabel("Epochs")
                plt.legend(loc=0)
                # plt.show()
                plt.grid()
                plt.savefig('plots/%i_summary_%s.png' % (last_idx, mode), dpi=300)


if __name__ == '__main__':
    plot_validation_summary('error')
