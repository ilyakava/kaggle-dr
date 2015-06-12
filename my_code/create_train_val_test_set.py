import sys
import csv
import random
import numpy

import pdb

def create_train_val_test_set(label_csv, valid_set_size, test_set_size, extension=".jpeg", K=5):
    # create set of int ids
    id_to_y = {} # just a reference
    y_to_id = {} # acts as our pool
    for k in xrange(K):
        y_to_id[k] = []

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
            test_dataset["%s%s" % (id, extension)] = id_to_y[id]
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
            valid_dataset["%s%s" % (id, extension)] = id_to_y[id]
            y_to_id[y].remove(id)

    train_dataset = {}
    for ids in y_to_id.values():
        for id in ids:
            train_dataset["%s%s" % (id, extension)] = id_to_y[id]

    print("Final Descending Proportions")
    print("All:   {}".format(proportions))
    test_proportions = [sum(numpy.array(test_dataset.values()) == klass)/float(len(test_dataset.values()) + numpy.spacing(1)) for klass in reversed(xrange(K))]
    print("Test:  {}".format(test_proportions))
    valid_proportions = [sum(numpy.array(valid_dataset.values()) == klass)/float(len(valid_dataset.values()) + numpy.spacing(1)) for klass in reversed(xrange(K))]
    print("Valid: {}".format(valid_proportions))
    train_proportions = [sum(numpy.array(train_dataset.values()) == klass)/float(len(train_dataset.values()) + numpy.spacing(1)) for klass in reversed(xrange(K))]
    print("Train: {}".format(train_proportions))

    return (train_dataset, valid_dataset, test_dataset)
