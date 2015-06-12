import os.path
import numpy
import theano
from skimage.io import imread

from create_train_val_test_set import create_train_val_test_set

import pdb

class DataStream(object):
    def __init__(self,
                 image_dir="data/train/centered_crop/",
                 image_shape=(128, 128, 3),
                 cache_size=1024,
                 center=0,
                 normalize=0,
                 amplify=1,
                 num_output_classes=5):
        self.image_dir = image_dir
        self.image_shape = image_shape
        self.cache_size = cache_size # size in images
        self.center = center
        self.mean = None
        self.normalize = normalize
        self.std = None
        self.amplify = amplify

        csv_dir = '/'.join(image_dir.split('/')[:-2] + ["trainLabels.csv"]) # csv should rest 1 dir up from image dir provided
        self.train_dataset, self.valid_dataset, test_dataset = create_train_val_test_set(csv_dir, valid_set_size=4864, test_set_size=0, extension=".png", K=num_output_classes)
        if self.center == 1 or self.normalize == 1:
            self.calc_mean_std_image()

    def valid_set(self):
        all_val_images = numpy.zeros(((len(self.valid_dataset),) + self.image_shape), dtype=theano.config.floatX)
        for i, image in enumerate(self.valid_dataset):
            all_val_images[i, ...] = self.feed_image(image) # b01c, Theano: bc01 CudaConvnet: c01b
        return numpy.rollaxis(all_val_images, 3, 1), numpy.array(self.valid_dataset.values(), dtype='int32')

    def train_buffer(self):
        """
        Yields a x_cache_block, has a size that is a multiple of training batches
        """
        x_cache_block = numpy.zeros(((self.cache_size,) + self.image_shape), dtype=theano.config.floatX)
        n_cache_blocks = int(len(self.train_dataset) / float(self.cache_size)) # rounding down skips the leftovers
        assert(n_cache_blocks)
        for ith_cache_block in xrange(n_cache_blocks):
            ith_cache_block_end = (ith_cache_block + 1) * self.cache_size
            ith_cache_block_slice = slice(ith_cache_block * self.cache_size, ith_cache_block_end)
            for i, image in enumerate(self.train_dataset.keys()[ith_cache_block_slice]):
                x_cache_block[i, ...] = self.feed_image(image)
            yield numpy.rollaxis(x_cache_block, 3, 1), numpy.array(self.train_dataset.values()[ith_cache_block_slice], dtype='int32')


    def read_image(self, image):
        """
        :type image: string
        """
        as_grey = True if self.image_shape[2] == 1 else False
        img = imread(self.image_dir + image, as_grey=as_grey)
        return (img.reshape(self.image_shape) / 255.) # reshape in case it is as_grey

    def preprocess_image(self, image):
        """
        Important, use with read_image. This method assumes image is already
        standardized to have [0,1] pixel values
        """
        if not self.mean == None:
            image = image - self.mean
        if not self.std == None:
            image = image / (self.std + 1e-5)
        return self.amplify * image

    def feed_image(self, image):
        img = self.read_image(image)
        return self.preprocess_image(img)

    def calc_mean_std_image(self):
        """
        Streaming variance calc: http://math.stackexchange.com/questions/20593/calculate-variance-from-a-stream-of-sample-values
        """
        print("Calculating mean and std dev image...")
        mean = numpy.zeros(self.image_shape, dtype=theano.config.floatX)
        mean_sqr = numpy.zeros(self.image_shape, dtype=theano.config.floatX)
        N = len(self.train_dataset)
        for image in self.train_dataset.keys():
            img = self.read_image(image)
            mean += img
            mean_sqr += numpy.square(img)
        self.mean = mean / N
        self.std = numpy.sqrt(numpy.abs(mean_sqr / N - numpy.square(self.mean)))
