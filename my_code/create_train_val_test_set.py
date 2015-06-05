import sys
import csv
import random
import numpy

from ciresan.code.package_data import package_data

import pdb

K = 5 # num classes

def create_train_val_set(image_directory, label_csv, valid_set_size, test_set_size, image_shape, outfile_path):
    # create set of int ids
    id_to_y = {} # just a reference
    y_to_id = {0:[], 1:[], 2:[], 3:[], 4:[]} # acts as our pool
    with open(label_csv, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            id = row[0]
            y = int(row[1])

            y_to_id[y].append(id)
            id_to_y[id] = int(row[1])

    total = float(len(id_to_y))
    proportions = [(len(y_to_id[klass]) / total) for klass in reversed(xrange(K))]
    print "descending class proportions are: {}".format(proportions)

    test_dataset = {}
    for i, proportion in enumerate(proportions):
        y = K - i - 1
        num_ys_wanted = int(proportion * test_set_size)
        if not y: # on most plentiful class 0, fill in the rest of the dataset
            num_ys_needed = test_set_size - len(test_dataset)
            print("taking an extra %i klass %i's into test set" % (num_ys_needed-num_ys_wanted, y))
            num_ys_wanted = num_ys_needed

        test_set_y_recruits = random.sample(y_to_id[y], num_ys_wanted)
        # add these to test set and remove from pool
        for id in test_set_y_recruits:
            test_dataset["%s.jpeg" % id] = id_to_y[id]
            y_to_id[y].remove(id)

    valid_dataset = {}
    for i, proportion in enumerate(proportions):
        y = K - i - 1
        num_ys_wanted = int(proportion * valid_set_size)
        if not y: # on most plentiful class 0, fill in the rest of the dataset
            num_ys_needed = valid_set_size - len(valid_dataset)
            print("taking an extra %i klass %i's into valid set" % (num_ys_needed-num_ys_wanted, y))
            num_ys_wanted = num_ys_needed

        test_set_y_recruits = random.sample(y_to_id[y], num_ys_wanted)
        # add these to test set and remove from pool
        for id in test_set_y_recruits:
            valid_dataset["%s.jpeg" % id] = id_to_y[id]
            y_to_id[y].remove(id)

    train_dataset = {}
    for ids in y_to_id.values():
        for id in ids:
            train_dataset["%s.jpeg" % id] = id_to_y[id]

    print("Final Descending Proportions")
    print("All:   {}".format(proportions))
    test_proportions = [sum(numpy.array(test_dataset.values()) == klass)/float(len(test_dataset.values())) for klass in reversed(xrange(K))]
    print("Test:  {}".format(test_proportions))
    valid_proportions = [sum(numpy.array(valid_dataset.values()) == klass)/float(len(valid_dataset.values())) for klass in reversed(xrange(K))]
    print("Valid: {}".format(valid_proportions))
    train_proportions = [sum(numpy.array(train_dataset.values()) == klass)/float(len(train_dataset.values())) for klass in reversed(xrange(K))]
    print("Train: {}".format(train_proportions))

    print("Pickling the Train/Valid/Test Partitions")
    package_data(image_directory, (train_dataset, valid_dataset, test_dataset), image_shape, outfile_path)

if __name__ == '__main__':
    arg_names = ['command', 'image_directory', 'outfile_path', 'valid_set_size', 'test_set_size', 'height', 'channels', 'label_csv']
    arg = dict(zip(arg_names, sys.argv))

    image_directory = arg.get('image_directory') or 'data/train/simple_crop/'
    outfile_path = arg.get('outfile_path') or 'data/train_testA.npz'
    valid_set_size = int(arg.get('valid_set_size') or 4864)
    test_set_size = int(arg.get('test_set_size') or 128)
    height = int(arg.get('height') or 112)
    channels = int(arg.get('channels') or 3)
    label_csv = arg.get('label_csv') or 'data/trainLabels.csv'

    image_shape = (height, height, channels)

    create_train_val_set(image_directory, label_csv, valid_set_size, test_set_size, image_shape, outfile_path)
