import sys
from os import listdir, path
import re
import cPickle
import numpy
import natsort

from my_code.predict import model_runid, save_prediction

import pdb

TEST_IMAGE_DIR = 'data/test/centered_crop/'

# usage:
if __name__ == '__main__':
    raw_output_files = sys.argv[1].split(',')
    raw_outputs = [cPickle.load(open(f)) for f in raw_output_files]
    avg = sum(raw_outputs) / float(len(raw_output_files))

    runids = [model_runid(n) for n in raw_output_files]
    name = '+'.join(runids)
    labels = (avg > 0.5).sum(axis=1)
    img_names = numpy.array([path.splitext(f)[0] for f in listdir(TEST_IMAGE_DIR) if re.search('\.(jpeg|png)', f)])
    save_prediction(name, natsort.natsorted(img_names), labels)
