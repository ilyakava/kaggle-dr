import unittest
import tempfile
import csv
import uuid
import numpy

from my_code.block_designer import BlockDesigner

import pdb

PROPORTION_ERROR_MARGIN = 0.01 # 1 percent
SAMPLE_COUNT_ERROR_MARGIN = 5 # misplaced samples
ACTUAL_TRAIN_DR_PROPORTIONS = [25810, 2443, 5292, 873, 708]

def create_skewed_CSV():
    populations = ACTUAL_TRAIN_DR_PROPORTIONS
    K = len(populations)
    proportions = [(populations[klass] / float(sum(populations))) for klass in reversed(xrange(K))]
    id = 0
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    with tmp as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image','level'])
        for klass, population in enumerate(populations):
            for _ in xrange(population):
                id += 1
                writer.writerow(['%s_%i' % (str(uuid.uuid4())[:8], id), klass])
    return(tmp.name, numpy.array(proportions))

def get_proportions(dataset):
    return numpy.array([len(dataset[klass])/(len(numpy.concatenate(dataset.values()).flatten()) + numpy.spacing(1)) for klass in reversed(sorted(dataset.keys()))])

class CreateTrainValTestSet(unittest.TestCase):

    def setUp(self):
        f, self.true_proportions = create_skewed_CSV()
        self.K = len(self.true_proportions)
        self.bd = BlockDesigner(f, self.K)

    def get_counts(self, dataset):
        return numpy.array([len(dataset[klass]) for klass in reversed(xrange(self.K))])

    def test_instantiating_and_splitting_multiple_times(self):
        valid_dataset = self.bd.break_off_block(4864)
        train_dataset = self.bd.remainder()
        train_batches_to_take = self.bd.size() // 128

        bd2 = BlockDesigner(train_dataset)
        batches2 = bd2.break_off_multiple_blocks(train_batches_to_take, 128)
        bd3 = BlockDesigner(train_dataset)
        batches3 = bd3.break_off_multiple_blocks(train_batches_to_take, 128)

        ideal_counts = numpy.array([int(128 * p) for p in self.true_proportions])

        for i in xrange(len(batches2)):
            counts = self.get_counts(batches2[i])
            self.failUnless(
                sum(counts) == 128
            )
            self.failUnless(
                sum(abs(self.get_counts(batches2[i]) - ideal_counts)) < SAMPLE_COUNT_ERROR_MARGIN
            )

            counts = self.get_counts(batches3[i])
            self.failUnless(
                sum(counts) == 128
            )
            self.failUnless(
                sum(abs(self.get_counts(batches3[i]) - ideal_counts)) < SAMPLE_COUNT_ERROR_MARGIN
            )

    def test_small_blocks_for_consistency(self):
        valid_dataset = self.bd.break_off_block(4864)

        bd2 = BlockDesigner(valid_dataset)
        batches = bd2.break_off_multiple_blocks(int(4864 / 128.), 128)

        ideal_counts = numpy.array([int(128 * p) for p in self.true_proportions])

        self.failUnless(
            bd2.size() == 0
        )
        for i in xrange(len(batches)):
            counts = self.get_counts(batches[i])
            self.failUnless(
                sum(counts) == 128
            )
            self.failUnless(
                sum(abs(self.get_counts(batches[i]) - ideal_counts)) < SAMPLE_COUNT_ERROR_MARGIN
            )

    def test_no_test_set(self):
        valid_dataset = self.bd.break_off_block(4864)
        train_dataset = self.bd.remainder()
        self.failUnless(
            sum(self.get_counts(valid_dataset) + self.get_counts(train_dataset)) == self.bd.init_size
        )

        valid_proportions = get_proportions(valid_dataset)
        train_proportions = get_proportions(train_dataset)

        self.failUnless(
            sum(abs(valid_proportions - self.true_proportions) < PROPORTION_ERROR_MARGIN) == self.K
        )
        self.failUnless(
            sum(abs(train_proportions - self.true_proportions) < PROPORTION_ERROR_MARGIN) == self.K
        )

    def test_all_sets(self):
        test_dataset = self.bd.break_off_block(1024)
        valid_dataset = self.bd.break_off_block(4864)
        train_dataset = self.bd.remainder()
        self.failUnless(
            sum(self.get_counts(test_dataset) + self.get_counts(valid_dataset) + self.get_counts(train_dataset)) == self.bd.init_size
        )

        test_proportions = get_proportions(test_dataset)
        valid_proportions = get_proportions(valid_dataset)
        train_proportions = get_proportions(train_dataset)

        self.failUnless(
            sum(abs(test_proportions - self.true_proportions) < PROPORTION_ERROR_MARGIN) == self.K
        )
        self.failUnless(
            sum(abs(valid_proportions - self.true_proportions) < PROPORTION_ERROR_MARGIN) == self.K
        )
        self.failUnless(
            sum(abs(train_proportions - self.true_proportions) < PROPORTION_ERROR_MARGIN) == self.K
        )

def main():
  unittest.main()

if __name__ == '__main__':
  main()
