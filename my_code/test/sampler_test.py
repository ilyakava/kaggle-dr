import unittest
import tempfile
import csv
import uuid
import numpy

from my_code.sampler import Sampler
from my_code.test.block_designer_test import create_skewed_CSV, ACTUAL_TRAIN_DR_PROPORTIONS, get_proportions, PROPORTION_ERROR_MARGIN
from my_code.block_designer import BlockDesigner

import pdb

class SampleTrainValTestSet(unittest.TestCase):

    def setUp(self):
        f, self.true_proportions = create_skewed_CSV()
        self.K = len(self.true_proportions)
        self.bd = BlockDesigner(f, self.K)
        self.samp = Sampler(self.bd.remainder())

    def test_all_classes(self):
        for test_klass in range(self.bd.K):
            X, y = self.samp.custom_distribution(test_klass, 128)
            self.failUnless(
                (len(y) == len(X)) and (len(X) % 128 == 0)
            )

    def test_cycles_through_all_data(self):
        X, y = self.samp.custom_distribution(0, 128)
        X2, y2 = self.samp.custom_distribution(0, 128)
        self.failUnless(
            len(set(X+X2)) == sum(ACTUAL_TRAIN_DR_PROPORTIONS)
        )

    def test_custom_distribution(self):
        X, y = self.samp.custom_distribution(0, 128, [94,9,19,3,3])

        collect = {}
        for k in set(y):
            collect[k] = []
        for i, klass in enumerate(y):
            collect[klass].append(X[i])

        self.failUnless(
            sum(abs(get_proportions(collect) - self.true_proportions) < PROPORTION_ERROR_MARGIN) == self.K
        )

    def test_skipping_classes(self):
        X, y = self.samp.custom_distribution(0, 128, [64,64,0,0,0])

        collect = {}
        for k in set(y):
            collect[k] = []
        for i, klass in enumerate(y):
            collect[klass].append(X[i])

        self.failUnless(
            sum(abs(get_proportions(collect) - numpy.array([0.5, 0.5])) < PROPORTION_ERROR_MARGIN) == 2
        )

def main():
  unittest.main()

if __name__ == '__main__':
  main()
