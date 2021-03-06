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

def downsize_no_trim_batchfile(files, size, in_dir, out_dir, index):
    name = "data/convert/downsize_no_trim_batchfile_%i.txt" % index
    out = open(name, 'w')
    for f in files:
        cmd = 'convert %s%s -scale %ix%i -quality 100 -normalize %s%s\n' % (in_dir, f, size, size, out_dir, f)
        out.write (cmd)
    out.close()
    return name

def centered_normalized_crop_batchfile(files, size, in_dir, out_dir, index):
    name = "data/convert/centered_normalized_crop_batchfile_%i.txt" % index
    out = open(name, 'w')
    for f_ in files:
        f = '.'.join([f_.split('.')[0], 'png'])
        cmd = 'convert %s%s -format png -normalize -bordercolor black -border 1x1 -fuzz 10%% -trim +repage -gravity center -resize %i -background black -gravity center -extent %ix%i %s%s\n' % (in_dir, f_, size, size, size, out_dir, f)
        out.write (cmd)
    out.close()
    return name

def resize_smaller_dim_batchfile(files, size, in_dir, out_dir, index):
    name = "data/convert/resize_smaller_dim_batchfile_%i.txt" % index
    out = open(name, 'w')
    for f_ in files:
        f = '.'.join([f_.split('.')[0], 'png'])
        three_chans = "-depth 8 -type TrueColor" # guarantees three color channels even if image is in grayscale
        cmd = 'convert %s%s -scale %ix%i^ -format png -quality 100 %s %s%s\n' % (in_dir, f_, size, size, three_chans, out_dir, f)
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

    # Note: performance varies wildly between machines:
    # http://stackoverflow.com/questions/30625635/can-graphicsmagick-batch-process-on-more-than-2-threads
    num_batch_files = int(arg.get('num_batch_files') or 2)

    files = numpy.array(listdir(in_dir))
    split_idx_multiple = len(files) / num_batch_files
    split_idxs = (numpy.arange(num_batch_files) + 1) * split_idx_multiple
    groups = numpy.split(files, split_idxs[:-1])

    if mode == 1:
        files = gm_batchfiles(groups, size, in_dir, out_dir, simple_crop_batchfile)
    elif mode == 2:
        files = gm_batchfiles(groups, size, in_dir, out_dir, centered_crop_batchfile)
    elif mode == 3:
        files = gm_batchfiles(groups, size, in_dir, out_dir, downsize_no_trim_batchfile)
    elif mode == 4:
        files = gm_batchfiles(groups, size, in_dir, out_dir, centered_normalized_crop_batchfile)
    elif mode == 5:
        files = gm_batchfiles(groups, size, in_dir, out_dir, resize_smaller_dim_batchfile)
    else:
        raise ValueError("unsupported mode %i" % mode)

    print 'done'
    print("You have %i commands to run:" % len(files))
    for filename in files:
        print "gm batch -echo on -feedback on %s" % filename
