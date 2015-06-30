import sys
import csv
import random
import numpy

import pdb

class BlockDesigner(object):
    """
    Serves batches with the same distribution of labels in each batch
    """
    def __init__(self, source, K=5, seed=None):
        """
        :type source: string or dict[int->list[str]]
        :param source: name of a csv or the output of a previous BlockDesigner.break_off_block

        Instance variables:

        :type self.reservoir: mutable dict[int->list[str]]
        :param self.reservoir: key is a y label, value is a image name

        :param proportions: acts as the design for each block that
            BlockDesigner serves
        """
        if seed:
            random.seed(seed)
        self.K = K
        self.reservoir = {} # will act as our pool that slowly drains, a source for blocks
        for k in xrange(self.K):
            self.reservoir[k] = []

        if type(source) is dict:
            self.fill_reservoir_with_dict(source)
        elif type(source) is str:
            self.fill_reservoir_with_csv(source)
        else:
            raise ValueError("unsupported data source: %s" % str(type(source)))

        self.reference = self.invert_reservoir()
        self.init_size = self.size()
        # We put the proportions in reverse order because the pathological observations
        # are substancialy less frequent (sever diagnosis -> higher class number)
        self.proportions = numpy.array([(len(self.reservoir[klass]) / float(self.init_size)) for klass in reversed(xrange(self.K))])

    def fill_reservoir_with_csv(self, label_csv):
        with open(label_csv, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)
            for row in reader:
                id = row[0]
                y = int(row[1])

                self.reservoir[y].append(id)

    def fill_reservoir_with_dict(self, source):
        assert(type(source.keys()[0]) is int)
        assert(type(source.values()[0]) is list)
        assert(type(source.values()[0][0]) is str)
        for y, ids in source.items():
            for id in ids:
                self.reservoir[y].append(id)

    def invert_reservoir(self):
        reference = {}
        for y, ids in self.reservoir.items():
            for id in ids:
                reference[id] = y
        return reference

    # an alias
    def remainder(self):
        return self.reservoir

    def size(self):
        return len(numpy.concatenate(self.reservoir.values()))

    def ids(self):
        return numpy.concatenate(self.reservoir.values())

    def break_off_multiple_blocks(self, num_blocks, block_size):
        """
        Mutates the reservoir that it picks the subset from: samples without replacement.

        :return: list of dictionaries of the same format as self.reservoir
        """
        if self.size() < block_size * num_blocks:
            raise ValueError("Requested %i examples when only %i available" % (block_size * num_blocks, self.size()))
        ideal_counts = [int(i) for i in (block_size * self.proportions)]
        num_random_additions = block_size - sum(ideal_counts)
        blocks = []
        for i in range(num_blocks):
            block = {}
            for y, count in enumerate(reversed(ideal_counts)):
                block[y] = random.sample(self.reservoir[y], count)
                for id in block[y]:
                    self.reservoir[y].remove(id)
            blocks.append(block)
        # a second separate loop for fill in, rather than being integrated into the above
        # loop, guarantees base ideal_counts in each partition
        for i in range(num_blocks):
            random_additions = random.sample(self.ids(), num_random_additions)
            for id in random_additions:
                y = self.reference[id]
                blocks[i][y] = blocks[i][y] + [id]
                self.reservoir[y].remove(id)
        return blocks

    def break_off_block(self, block_size):
        """
        Mutates the reservoir that it picks the subset from: samples without replacement.

        :return: a dictionary of the same format as self.reservoir
        """
        if block_size < 1024:
            print("[WARNING]: requested %i block_size is small, round off erros may impact final proportions")
        if self.size() < block_size:
            raise ValueError("Requested %i examples when only %i available" % (block_size, self.size()))
        K = len(self.proportions)
        subset = {}
        for k in xrange(K):
            subset[k] = []
        for i, proportion in enumerate(self.proportions):
            y = K - i - 1

            if y: # on rarer pathological cases
                exact_num_ys_wanted = proportion * block_size
                num_ys_wanted = int(proportion * block_size)
            else: # on most plentiful class 0, fill in the rest of the dataset
                num_ys_needed = block_size - len(numpy.concatenate(subset.values()))
                num_ys_wanted = num_ys_needed
                # This here is the reason that this method should only be used to
                # break off single large blocks. If this method is used repeatedly
                # then the proportions of the reservoir will become unbalanced. Attempts
                # were made earlier to compensate with random ways to select 'fill in'
                # example classes, but in the end, the best strategy is to have an
                # additional break_off_multiple_blocks method

            if not len(self.reservoir[y]):
                print("[WARNING]: Requested %i samples for class %i when NONE available" % (num_ys_wanted, y))
            if len(self.reservoir[y]) < num_ys_wanted:
                subset[y] = self.reservoir[y]
            else:
                subset[y] = random.sample(self.reservoir[y], num_ys_wanted)
            # batch remove from reservoir
            for id in subset[y]:
                self.reservoir[y].remove(id)
        return subset
