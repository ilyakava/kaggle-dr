import unittest
import tempfile
import csv
import uuid
import numpy

from my_code.block_designer import BlockDesigner

import pdb

PROPORTION_ERROR_MARGIN = 0.01 # 1 percent
SAMPLE_COUNT_ERROR_MARGIN = 5 # misplaced samples

class CreateTrainValTestSet(unittest.TestCase):

    def setUp(self):
        self.bd, self.true_proportions = self.create_skewed_CSV()

    def create_skewed_CSV(self):
        populations = [25810, 2443, 5292, 873, 708] # actual DR proportions in descending classes
        self.K = len(populations)
        proportions = [(populations[klass] / float(sum(populations))) for klass in reversed(xrange(self.K))]
        id = 0
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        with tmp as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image','level'])
            for klass, population in enumerate(populations):
                for _ in xrange(population):
                    id += 1
                    writer.writerow(['%s_%i' % (str(uuid.uuid4())[:8], id), klass])
        return BlockDesigner(tmp.name, self.K), numpy.array(proportions)

    def get_proportions(self, dataset):
        return numpy.array([len(dataset[klass])/(len(numpy.concatenate(dataset.values()).flatten()) + numpy.spacing(1)) for klass in reversed(xrange(self.K))])

    def get_counts(self, dataset):
        return numpy.array([len(dataset[klass]) for klass in reversed(xrange(self.K))])

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

        K = len(self.true_proportions)
        valid_proportions = self.get_proportions(valid_dataset)
        train_proportions = self.get_proportions(train_dataset)

        self.failUnless(
            sum(abs(valid_proportions - self.true_proportions) < PROPORTION_ERROR_MARGIN) == K
        )
        self.failUnless(
            sum(abs(train_proportions - self.true_proportions) < PROPORTION_ERROR_MARGIN) == K
        )

    def test_all_sets(self):
        test_dataset = self.bd.break_off_block(1024)
        valid_dataset = self.bd.break_off_block(4864)
        train_dataset = self.bd.remainder()
        self.failUnless(
            sum(self.get_counts(test_dataset) + self.get_counts(valid_dataset) + self.get_counts(train_dataset)) == self.bd.init_size
        )

        K = len(self.true_proportions)
        test_proportions = self.get_proportions(test_dataset)
        valid_proportions = self.get_proportions(valid_dataset)
        train_proportions = self.get_proportions(train_dataset)

        self.failUnless(
            sum(abs(test_proportions - self.true_proportions) < PROPORTION_ERROR_MARGIN) == K
        )
        self.failUnless(
            sum(abs(valid_proportions - self.true_proportions) < PROPORTION_ERROR_MARGIN) == K
        )
        self.failUnless(
            sum(abs(train_proportions - self.true_proportions) < PROPORTION_ERROR_MARGIN) == K
        )

def main():
  unittest.main()

if __name__ == '__main__':
  main()
