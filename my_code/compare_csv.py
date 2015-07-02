import sys
import csv
import numpy
import pandas
from my_code.predict_util import QWK, print_confusion_matrix

import pdb

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
    kappa, M = QWK(numpy.array(y1),numpy.array(y2))
    print "Kappa = %.5f" % kappa
    print_confusion_matrix(M)
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
