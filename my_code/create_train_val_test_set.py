import sys
import csv
import random

from ciresan.code.package_data import package_data

import pdb

def create_train_val_set(image_directory, label_csv, val_set_size, test_set_size, image_shape, outfile_path):
    assert(test_set_size % 2 == 0)
    assert(val_set_size % 2 == 0)
    # create set of int ids
    entire_dataset = {}
    ids = set()
    with open(label_csv, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            filename = row[0]
            ids.add(int(filename.split('_')[0]))
            entire_dataset[filename] = int(row[1])

    # choose test_set_size/2 of them
    test_set_ids = random.sample(ids, test_set_size / 2)
    # create the dictionaries
    test_dataset = {}
    for test_id in test_set_ids:
        ids.remove(test_id)

        left = "%i_left" % test_id
        right = "%i_right" % test_id

        test_dataset["%s.jpeg" % left] = entire_dataset[left]
        test_dataset["%s.jpeg" % right] = entire_dataset[right]

    val_set_ids = random.sample(ids, val_set_size / 2)
    val_dataset = {}
    for val_id in val_set_ids:
        ids.remove(val_id)

        left = "%i_left" % val_id
        right = "%i_right" % val_id

        val_dataset["%s.jpeg" % left] = entire_dataset[left]
        val_dataset["%s.jpeg" % right] = entire_dataset[right]

    train_dataset = {}
    for train_id in ids:
        left = "%i_left" % train_id
        right = "%i_right" % train_id

        train_dataset["%s.jpeg" % left] = entire_dataset[left]
        train_dataset["%s.jpeg" % right] = entire_dataset[right]
    # pickle the selection
    package_data(image_directory, (train_dataset, val_dataset, test_dataset), image_shape, outfile_path)

if __name__ == '__main__':
    arg_names = ['command', 'image_directory', 'outfile_path', 'val_set_size', 'test_set_size', 'height', 'channels', 'label_csv']
    arg = dict(zip(arg_names, sys.argv))

    image_directory = arg.get('image_directory') or 'data/train/simple_crop/'
    outfile_path = arg.get('outfile_path') or 'data/train_testA.npz'
    val_set_size = int(arg.get('val_set_size') or 1406)
    test_set_size = int(arg.get('test_set_size') or 100)
    height = int(arg.get('height') or 112)
    channels = int(arg.get('channels') or 3)
    label_csv = arg.get('label_csv') or 'data/trainLabels.csv'

    image_shape = (height, height, channels)

    create_train_val_set(image_directory, label_csv, val_set_size, test_set_size, image_shape, outfile_path)
