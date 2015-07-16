import random
import numpy
import itertools
from my_code.block_designer import BlockDesigner

import pdb

class Sampler(BlockDesigner):
    """
    Usefully inherits __init__ so that it can be initialized with
    a block from a BlockDesigner
    """
    def uniform_full_sample_class(self, k, batch_size):
        """
        loops through all the examples in class k filling in with other
        classes as neccessary. Classes > k will be oversampled and classes
        < k will be undersampled
        """
        all_klasses = self.reservoir.keys()
        klass_examples = self.reservoir[k]
        random.shuffle(klass_examples)
        for klass in all_klasses:
            random.shuffle(self.reservoir[klass])
        loopy_klass_examples = [itertools.cycle(self.reservoir[k_]) for k_ in all_klasses]

        avg_num_batch_klass_examples = int(batch_size / self.K)
        num_extras_needed = batch_size - avg_num_batch_klass_examples * self.K
        extra_pad = numpy.zeros(self.K, dtype=int)
        extra_pad[:num_extras_needed] = 1 # take 1 extra of most prevalent classes
        num_all_klass_examples = numpy.array([avg_num_batch_klass_examples] * self.K, dtype=int) + extra_pad
        assert(sum(num_all_klass_examples) == batch_size)

        num_klass_examples = num_all_klass_examples[k]
        num_batches_to_yield = int(len(klass_examples) / num_all_klass_examples[k])

        X = []
        y = []
        for batch_i in range(num_batches_to_yield):
            X_batch = []
            y_batch = []
            klass_slice = slice(batch_i*num_klass_examples, (batch_i+1)*num_klass_examples)
            X_batch.extend(klass_examples[klass_slice])
            y_batch.extend((k*numpy.ones(num_klass_examples)).tolist())
            # fill in the rest
            for klass, num_examples in enumerate(num_all_klass_examples):
                if not klass == k:
                    for i in range(num_examples):
                        X_batch.append(loopy_klass_examples[klass].next())
                    y_batch.extend((klass*numpy.ones(num_examples)).tolist())
            assert(len(X_batch) == batch_size)
            assert(len(y_batch) == batch_size)
            X.extend(X_batch)
            y.extend(y_batch)
        return(X, y)
