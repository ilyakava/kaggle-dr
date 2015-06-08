# -*- coding: utf-8 -*-
import sys
from os import listdir
import numpy

import pdb

def gm_batchfiles(groups, size, in_dir, out_dir, func):
    # creates on batchfile
    outfiles = []
    for index, group in enumerate(groups):
        outfiles.append(func(group, size, in_dir, out_dir, index))
    return outfiles

def simple_crop_batchfile(files, size, in_dir, out_dir, index):
    name = "data/convert/simple_crop_batchfile_%i.txt" % index
    out = open(name, 'w')
    for f in files:
        cmd = 'convert %s%s -fuzz 10%% -trim -scale %ix%i^ -gravity center -extent %ix%i -quality 100 %s%s\n' % (in_dir, f, size, size, size, size, out_dir, f)
        out.write (cmd)
    out.close()
    return name

def centered_crop_batchfile(files, size, in_dir, out_dir, index):
    name = "data/convert/centered_crop_batchfile_%i.txt" % index
    out = open(name, 'w')
    for f_ in files:
        f = '.'.join([f_.split('.')[0], 'png'])
        cmd = 'convert %s%s -format png -bordercolor black -border 1x1 -fuzz 10%% -trim +repage -gravity center -resize %i -background black -gravity center -extent %ix%i %s%s\n' % (in_dir, f_, size, size, size, out_dir, f)
        out.write (cmd)
    out.close()
    return name

if __name__ == '__main__':
    arg_names = ['command', 'in_dir', 'out_dir', 'mode', 'size', 'num_batch_files']
    arg = dict(zip(arg_names, sys.argv))

    in_dir = arg.get('in_dir') or 'data/train/orig/'
    out_dir = arg.get('out_dir') or 'data/train/simple_crop/'
    mode = int(arg.get('mode') or 1)
    size = int(arg.get('size') or 112)
    # graphicsmagick seems to only run on 1 core, there is a 30% increase in performance
    # using 2 over 1 batch files (probably just b/c of less idleness)
    # TODO: try forking from python and doing processing from this script
    num_batch_files = int(arg.get('num_batch_files') or 2)

    files = numpy.array(listdir(in_dir))
    split_idx_multiple = len(files) / num_batch_files
    split_idxs = (numpy.arange(num_batch_files) + 1) * split_idx_multiple
    groups = numpy.split(files, split_idxs[:-1])

    if mode == 1:
        print gm_batchfiles(groups, size, in_dir, out_dir, simple_crop_batchfile)
    elif mode == 2:
        print gm_batchfiles(groups, size, in_dir, out_dir, centered_crop_batchfile)
    else:
        raise ValueError("unsupported mode %i" % mode)

    print 'done'
    print 'run: gm batch -echo on -feedback on XXX'
