from os import listdir, path
import random
import csv
import re
import natsort
import numpy
import theano
from skimage.io import imread

from block_designer import BlockDesigner

import pdb

def no_flip(image_name):
    return [0,0]

def rand_flip(image_name):
    return [int(round(random.random())), int(round(random.random()))]

def get_train_flip_lambda(lambda_or_file_name):
    flip_lambdas = {
        "no_flip": no_flip,
        "rand_flip": rand_flip
    }
    if flip_lambdas.get(lambda_or_file_name):
        return flip_lambdas[lambda_or_file_name]
    else:
        image_name_to_flip_coord = {}
        with open(lambda_or_file_name, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)
            for row in reader:
                image_name = row[0]
                flip_coords = [int(row[1]), int(row[2])]
                image_name_to_flip_coord[image_name] = flip_coords
        def predetermined_flip(image_name):
            return image_name_to_flip_coord[image_name]
        return predetermined_flip

def get_train_val_flip_lambda(lambda_or_file_name):
    train = get_train_flip_lambda(lambda_or_file_name)
    val = train if re.search('\.csv', lambda_or_file_name) else no_flip
    return(train,val)

TRAIN_LABELS_CSV_PATH = "data/train/trainLabels.csv"

class DataStream(object):
    """
    Provides an interface for easily filling and replacing GPU cache of images
    """
    def __init__(self,
                 train_image_dir="data/train/centered_crop/",
                 image_shape=(128, 128, 3),
                 cache_size=1024,
                 batch_size=128,
                 center=0,
                 normalize=0,
                 amplify=1,
                 train_flip='no_flip',
                 shuffle=1,
                 test_image_dir=None,
                 random_seed=None,
                 valid_dataset_size=4864):
        self.train_image_dir = train_image_dir
        self.test_image_dir = test_image_dir
        self.image_shape = image_shape
        self.cache_size = cache_size # size in images
        self.batch_size = batch_size
        self.center = center
        self.mean = None
        self.normalize = normalize
        self.std = None
        self.amplify = amplify
        self.train_flip_lambda, self.val_flip_lambda = get_train_val_flip_lambda(train_flip)
        self.valid_dataset_size = valid_dataset_size

        bd = BlockDesigner(TRAIN_LABELS_CSV_PATH, seed=random_seed)

        valid_examples = bd.break_off_block(self.valid_dataset_size)
        self.train_examples = bd.remainder()
        self.n_train_batches = int(bd.size() / self.batch_size)
        self.train_dataset_size = self.n_train_batches * self.batch_size

        self.valid_dataset = self.setup_valid_dataset(valid_examples)
        self.train_dataset = None if shuffle else self.setup_train_dataset()
        self.test_dataset = self.setup_test_dataset()
        self.n_test_examples = len(self.test_dataset["X"])

        if self.center == 1 or self.normalize == 1:
            self.calc_mean_std_image()

    def valid_set(self):
        all_val_images = numpy.zeros(((len(self.valid_dataset["y"]),) + self.image_shape), dtype=theano.config.floatX)
        for i, image in enumerate(self.valid_dataset["X"]):
            all_val_images[i, ...] = self.feed_image(image, self.train_image_dir, self.val_flip_lambda) # b01c, Theano: bc01 CudaConvnet: c01b
        return numpy.rollaxis(all_val_images, 3, 1), numpy.array(self.valid_dataset["y"], dtype='int32')

    def train_buffer(self):
        """
        Yields a x_cache_block, has a size that is a multiple of training batches
        """
        train_dataset = self.train_dataset or self.setup_train_dataset()
        x_cache_block = numpy.zeros(((self.cache_size,) + self.image_shape), dtype=theano.config.floatX)
        n_cache_blocks = int(len(train_dataset["y"]) / float(self.cache_size)) # rounding down skips the leftovers
        if not n_cache_blocks:
            raise ValueError("Train dataset length %i is too small for cache size %i" % (len(train_dataset["y"]), self.cache_size))
        for ith_cache_block in xrange(n_cache_blocks):
            ith_cache_block_end = (ith_cache_block + 1) * self.cache_size
            ith_cache_block_slice = slice(ith_cache_block * self.cache_size, ith_cache_block_end)
            for i, image in enumerate(train_dataset["X"][ith_cache_block_slice]):
                x_cache_block[i, ...] = self.feed_image(image, self.train_image_dir, self.train_flip_lambda)
            yield numpy.rollaxis(x_cache_block, 3, 1), numpy.array(train_dataset["y"][ith_cache_block_slice], dtype='int32')

    def test_buffer(self):
        """
        Yields a x_cache_block, has a size that is a multiple of training batches
        """
        x_cache_block = numpy.zeros(((self.cache_size,) + self.image_shape), dtype=theano.config.floatX)
        n_full_cache_blocks, n_leftovers = divmod(len(self.test_dataset["X"]), self.cache_size)
        if not n_full_cache_blocks:
            raise ValueError("Test dataset length %i is too small for cache size %i" % (len(self.test_dataset["X"]), self.cache_size))
        for ith_cache_block in xrange(n_full_cache_blocks):
            ith_cache_block_end = (ith_cache_block + 1) * self.cache_size
            ith_cache_block_slice = slice(ith_cache_block * self.cache_size, ith_cache_block_end)
            idxs_to_full_dataset = list(range(ith_cache_block * self.cache_size, ith_cache_block_end))
            for i, image in enumerate(self.test_dataset["X"][ith_cache_block_slice]):
                x_cache_block[i, ...] = self.feed_image(image, self.test_image_dir, self.val_flip_lambda)
            yield numpy.rollaxis(x_cache_block, 3, 1), numpy.array(idxs_to_full_dataset, dtype='int32')
        # sneak the leftovers out, padded by the previous full cache block
        if n_leftovers:
            leftover_slice = slice(ith_cache_block_end, ith_cache_block_end + n_leftovers)
            for i, image in enumerate(self.test_dataset["X"][leftover_slice]):
                idxs_to_full_dataset[i] = ith_cache_block_end + i
                x_cache_block[i, ...] = self.feed_image(image, self.test_image_dir, self.val_flip_lambda)
            yield numpy.rollaxis(x_cache_block, 3, 1), numpy.array(idxs_to_full_dataset, dtype='int32')

    def read_image(self, image_name, image_dir, extension=".png"):
        """
        :type image: string
        """
        as_grey = True if self.image_shape[2] == 1 else False
        img = imread(image_dir + image_name + extension, as_grey=as_grey)
        return (img.reshape(self.image_shape) / 255.) # reshape in case it is as_grey

    def preprocess_image(self, image, flip_coords):
        """
        Important, use with read_image. This method assumes image is already
        standardized to have [0,1] pixel values
        """
        image = self.flip_image(image, flip_coords)
        if not self.mean == None:
            image = image - self.mean
        if not self.std == None:
            image = image / (self.std + 1e-5)
        return self.amplify * image

    def flip_image(self, image, flip_coords):
        if flip_coords[0] == 1:
            image = numpy.flipud(image)
        if flip_coords[1] == 1:
            image = numpy.fliplr(image)
        return image

    def feed_image(self, image_name, image_dir, flip_lambda=no_flip):
        img = self.read_image(image_name, image_dir)
        flip_coords = flip_lambda(image_name)
        return self.preprocess_image(img, flip_coords)

    def calc_mean_std_image(self):
        """
        Streaming variance calc: http://math.stackexchange.com/questions/20593/calculate-variance-from-a-stream-of-sample-values
        Will not look at the validation set images
        """
        print("Calculating mean and std dev image...")
        mean = numpy.zeros(self.image_shape, dtype=theano.config.floatX)
        mean_sqr = numpy.zeros(self.image_shape, dtype=theano.config.floatX)
        N = sum([len(ids) for y, ids in self.train_examples.items()]) # self.train_dataset_size + remainders
        for y, ids in self.train_examples.items():
            for image in ids:
                img = self.read_image(image, self.train_image_dir)
                mean += img
                mean_sqr += numpy.square(img)
        self.mean = mean / N
        self.std = numpy.sqrt(numpy.abs(mean_sqr / N - numpy.square(self.mean)))

    def setup_valid_dataset(self, block):
        images = []
        labels = []
        for y, ids in block.items():
            for id in ids:
                images.append(id)
                labels.append(y)
        return {"X": images, "y": labels}

    def setup_train_dataset(self):
        """
        Each self.batch_size of examples follows the same distribution
        """
        bd = BlockDesigner(self.train_examples)
        blocks = bd.break_off_multiple_blocks(self.n_train_batches, self.batch_size)
        images = []
        labels = []
        for block in blocks:
            for y, ids in block.items():
                for id in ids:
                    images.append(id)
                    labels.append(y)
        return {"X": images, "y": labels}

    def setup_test_dataset(self):
        if self.test_image_dir:
            images = numpy.array([path.splitext(f)[0] for f in listdir(self.test_image_dir) if re.search('\.(jpeg|png)', f)])
        else:
            images = []
        return {"X": natsort.natsorted(images)}
