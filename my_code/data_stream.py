import os.path
import numpy
import theano
from skimage.io import imread

from create_train_val_test_set import create_train_val_test_set

class DataStream(object):
    def __init__(self,
                 image_dir="data/train/centered_crop/",
                 image_shape=(128, 128, 3),
                 cache_size=1024):
        self.image_dir = image_dir
        self.image_shape = image_shape
        self.cache_size = cache_size # size in images

        self.train_dataset, self.valid_dataset, test_dataset = create_train_val_test_set("data/trainLabels.csv", valid_set_size=4864, test_set_size=0, extension=".png")

    def valid_set(self):
        all_val_images = numpy.zeros(((len(self.valid_dataset),) + self.image_shape), dtype=numpy.float32)
        for i, image in enumerate(self.valid_dataset):
            all_val_images[i, ...] = imread(self.image_dir + image)
        return numpy.rollaxis(all_val_images, 3, 1), numpy.array(self.valid_dataset.values(), dtype=numpy.int32)[numpy.newaxis].T

    def train_buffer(self):
        """
        Yields a x_cache_block, has a size that is a multiple of training batches
        """
        x_cache_block = numpy.zeros(((self.cache_size,) + self.image_shape), dtype=numpy.float32)
        n_cache_blocks = int(len(self.train_dataset) / float(self.cache_size)) # rounding down skips the leftovers
        assert(n_cache_blocks)
        for ith_cache_block in xrange(n_cache_blocks):
            ith_cache_block_end = (ith_cache_block + 1) * self.cache_size
            ith_cache_block_slice = slice(ith_cache_block * self.cache_size, ith_cache_block_end)
            for i, image in enumerate(self.train_dataset.keys()[ith_cache_block_slice]):
                x_cache_block[i, ...] = imread(self.image_dir + image)
            yield numpy.rollaxis(x_cache_block, 3, 1), numpy.array(self.train_dataset.values()[ith_cache_block_slice], dtype=numpy.int32)[numpy.newaxis].T
            # do we need to close the images?
