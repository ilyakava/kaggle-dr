import sys
import csv
import numpy
import pandas
from my_code.predict_util import QWK, print_confusion_matrix

import matplotlib
matplotlib.use('Agg')
from skimage.io import imread
matplotlib.rcParams.update({'font.size': 12})
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

import pdb

TAGS = ['woman', 'horse', 'hand', 'flower', 'bird', 'mountain', 'house', 'circle', 'tree', 'car']

def plot_confusion_matrix(M, labels=TAGS, outpath='plots/conf.png'):
    plt.imshow(M, interpolation='nearest', cmap=plt.cm.Greys)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = numpy.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label (Accuracy: %.4f)' % (numpy.diag(M).sum() / float(M.sum())))
    plt.savefig(outpath, dpi=600, bbox_inches='tight')

def csv_agreement(file1, file2):
    f1 = pandas.read_csv(file1)
    d1 = dict(zip(f1.image, f1.level))
    f2 = pandas.read_csv(file2)
    d2 = dict(zip(f2.image, f2.level))
    overlap_images = set(f1.image).intersection(set(f2.image))
    overlaps = {}
    y1 = []
    y2 = []
    for image in overlap_images:
        y1.append(d1[image])
        y2.append(d2[image])
        overlaps[image] = [d1[image], d2[image]]
    K = max(numpy.array(y1).max(), numpy.array(y2).max()) + 1
    kappa, M = QWK(numpy.array(y1),numpy.array(y2), K)
    print "Kappa = %.5f" % kappa
    print "Accuracy = %.5f" % (numpy.diag(M).sum() / float(M.sum()))
    print_confusion_matrix(M)
    plot_confusion_matrix(M)
    return overlaps

if __name__ == '__main__':
    assert(len(sys.argv) >= 3)
    overlaps = csv_agreement(sys.argv[1], sys.argv[2])
    if len(sys.argv) == 4:
        outfile_name = sys.argv[3]
        with open(outfile_name, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image','level_file1','level_file2'])
            for image, levels in overlaps.items():
                writer.writerow([image] + levels)
        print("Saved overlaps in: %s" % outfile_name)
