import sys
import numpy
import pandas
from my_code.predict_util import QWK, print_confusion_matrix

import pdb

def csv_agreement(file1, file2):
    f1 = pandas.read_csv(file1)
    d1 = dict(zip(f1.image, f1.level))
    f2 = pandas.read_csv(file2)
    d2 = dict(zip(f2.image, f2.level))
    overlap = set(f1.image).intersection(set(f2.image))
    y1 = []
    y2 = []
    for image in overlap:
        y1.append(d1[image])
        y2.append(d2[image])
    kappa, M = QWK(numpy.array(y1),numpy.array(y2))
    print "Kappa = %.5f" % kappa
    print_confusion_matrix(M)

if __name__ == '__main__':
    assert(len(sys.argv) == 3)
    csv_agreement(sys.argv[1], sys.argv[2])
