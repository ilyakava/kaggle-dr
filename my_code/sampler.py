import random
import numpy
import itertools
from my_code.block_designer import BlockDesigner

import pdb

class Sampler(BlockDesigner):
    """
    Inherits __init__ so that it can be initialized with
    a block from a BlockDesigner
    """
    def __init__(self, source, K=5, seed=None):
        super(Sampler, self).__init__(source, K=5, seed=None)
        all_klasses = self.reservoir.keys()
        self.loopy_klass_examples = [itertools.cycle(self.reservoir[k_]) for k_ in all_klasses]

    def uniform_klass_counts(self, batch_size):
        avg_num_batch_klass_examples = int(batch_size / self.K)
        num_extras_needed = batch_size - avg_num_batch_klass_examples * self.K
        extra_pad = numpy.zeros(self.K, dtype=int)
        extra_pad[:num_extras_needed] = 1 # take 1 extra of most prevalent classes
        all_num_feed_klass_examples = numpy.array([avg_num_batch_klass_examples] * self.K, dtype=int) + extra_pad
        return all_num_feed_klass_examples

    def custom_distribution(self, k, batch_size, requested_distribution=None):
        """
        loops through all the examples in class k filling in with other
        classes as neccessary. Assuming the default uniform distribution:
        classes greater than k will be oversampled (empirically leads to
        overfitting) and classes less than k will be undersampled
        """
        all_num_feed_klass_examples = requested_distribution or self.uniform_klass_counts(batch_size)
        assert(len(all_num_feed_klass_examples) == self.K)
        assert(sum(all_num_feed_klass_examples) == batch_size)

        num_feed_klass_examples = all_num_feed_klass_examples[k]
        num_batches_to_yield = int(len(self.reservoir[k]) / all_num_feed_klass_examples[k])

        X = []
        y = []
        for batch_i in range(num_batches_to_yield):
            X_batch = []
            y_batch = []

            for klass, num_examples in enumerate(all_num_feed_klass_examples):
                for _ in range(num_examples):
                    X_batch.append(self.loopy_klass_examples[klass].next())
                y_batch.extend((klass*numpy.ones(num_examples, dtype=int)).tolist())

            X.extend(X_batch)
            y.extend(y_batch)
        return(X, y)
