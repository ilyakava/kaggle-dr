import unittest
import tempfile
import csv
import uuid
import numpy

from my_code.sampler import Sampler
from my_code.test.block_designer_test import create_skewed_CSV
from my_code.block_designer import BlockDesigner

import pdb

class SampleTrainValTestSet(unittest.TestCase):

    def setUp(self):
        f, self.true_proportions = create_skewed_CSV()
        self.K = len(self.true_proportions)
        self.bd = BlockDesigner(f, self.K)

    def test_all_classes(self):
        samp = Sampler(self.bd.remainder())
        for test_klass in range(self.bd.K):
            X, y = samp.uniform_full_sample_class(test_klass, 128)
            self.failUnless(
                (len(y) == len(X)) and (len(X) % 128 == 0)
            )

def main():
  unittest.main()

if __name__ == '__main__':
  main()
