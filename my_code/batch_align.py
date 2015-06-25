import sys
from os import listdir
import numpy
from my_code.align_util import FundusPhotoAligner

import pdb

if __name__ == '__main__':
    arg_names = ['command', 'in_dir', 'num_total_batches', 'batch_num']
    arg = dict(zip(arg_names, sys.argv))

    in_dir = arg.get('in_dir') or 'data/train/orig/'
    num_total_batches = int(arg.get('num_total_batches') or 1)
    batch_num = int(arg.get('batch_num') or 1)

    assert((num_total_batches == 1) or (num_total_batches % 2 == 0))

    files = numpy.array([in_dir + f for f in listdir(in_dir)])
    split_idx_multiple = len(files) / num_total_batches
    split_idxs = (numpy.arange(num_total_batches) + 1) * split_idx_multiple
    groups = numpy.split(files, split_idxs[:-1])

    print 'Running batch %i of %i' % (batch_num, num_total_batches)
    cur_batch = groups[batch_num-1]

    fpa = FundusPhotoAligner(cur_batch)

    print 'Done batch %i of %i' % (batch_num, num_total_batches)
