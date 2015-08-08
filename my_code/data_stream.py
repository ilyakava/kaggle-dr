from os import listdir, path
import random
import csv
import re
import natsort
import numpy
import theano
from skimage.io import imread

from block_designer import BlockDesigner
from sampler import Sampler

import pdb

class ImageFlipOracle(object):
    """
    *_flip methods should take an image_name
    """
    def __init__(self, flip_mode):
        self.noise = 0
        if re.search('\.csv', flip_mode):
            self.image_name_to_flip_coord = {}
            with open(flip_mode, 'rb') as csvfile:
                reader = csv.reader(csvfile)
                next(reader, None)
                for row in reader:
                    image_name = row[0]
                    flip_coords = [int(row[1]), int(row[2])]
                    self.image_name_to_flip_coord[image_name] = flip_coords

    def get_flip_lambda(self, flip_mode, deterministic=False):
        if re.search('\.csv', flip_mode):
            if deterministic:
                return self.align_flip
            else:
                return self.noisy_align_flip
        else:
            return {
                "no_flip": self.no_flip,
                "rand_flip": self.rand_flip,
                "align_flip": self.align_flip,
                "noisy_align_flip": self.noisy_align_flip
            }[flip_mode]

    def no_flip(self, image_name):
        return numpy.zeros(2)

    def rand_flip(self, image_name):
        return numpy.array([int(round(random.random())), int(round(random.random()))])

    def align_flip(self, image_name):
        return numpy.array(self.image_name_to_flip_coord[image_name])

    def noisy_align_flip(self, image_name):
        """
        :param noise: float (0,1) where 1 is fully noise and 0 is
            fully deterministic. If greater than 0, predetermined correct flips
            will be swapped with a random flip with Pr(noise)
        """
        if random.random() < self.noise:
            return ((self.align_flip(image_name) + self.rand_flip(image_name)) % 2)
        else:
            return self.align_flip(image_name)

    def reset_noise(self, level):
        assert(level >= 0 and level <= 1)
        self.noise = level

class CropOracle(object):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def bottom_right_crop(self, img):
        h,w,c = img.shape
        max_t = h - self.out_dim
        max_l = w - self.out_dim
        return(max_t,h, max_l,w)

    def center_crop(self, img):
        max_t,h, max_l,w = self.bottom_right_crop(img)
        center_t = max_t / 2
        center_l = max_l / 2
        return(center_t, center_t + self.out_dim, center_l, center_l + self.out_dim)

    def uniform_crop(self, img):
        max_t,h, max_l,w = self.bottom_right_crop(img)
        rand_t = random.randint(0, max_t)
        rand_l = random.randint(0, max_l)
        return(rand_t, rand_t + self.out_dim, rand_l, rand_l + self.out_dim)

    def get_crop_lambda(self, mode):
        return {
            "center_crop": self.center_crop,
            "uniform_crop": self.uniform_crop
        }[mode]

class ColorCastOracle(object):
    def __init__(self, n_channels, color_cast_range):
        self.n_channels = n_channels
        self.color_cast_range = color_cast_range

    def no_cast(self):
        return numpy.zeros(self.n_channels)

    def baidu_cast(self):
        # http://arxiv.org/abs/1501.02876v3
        s = self.color_cast_range / 3.0 # 99.73% of values within 3 std deviations
        casts = []
        mask = []
        while len(casts) < self.n_channels:
            casts.append(numpy.random.normal(scale=s))
            mask.append(round(random.random()))
        return(numpy.array(casts, dtype=int) * numpy.array(mask, dtype=int))

    def get_color_cast_lambda(self, mode):
        return {
            "no_cast": self.no_cast,
            "baidu_cast": self.baidu_cast
        }[mode]

class DataStream(object):
    """
    Provides an interface for easily filling and replacing GPU cache of images
    """
    def __init__(self,
                 train_image_dir="data/train/centered_crop/",
                 image_shape=(128, 128, 3),
                 batch_size=128,
                 cache_size_factor=8,
                 center=0,
                 normalize=0,
                 amplify=1,
                 train_flip='no_flip',
                 shuffle=1,
                 test_image_dir=None,
                 random_seed=None,
                 valid_dataset_size=4864,
                 valid_flip='no_flip',
                 test_flip='no_flip',
                 sample_class=None,
                 custom_distribution=None,
                 train_color_cast='no_cast',
                 valid_color_cast='no_cast',
                 test_color_cast='no_cast',
                 color_cast_range=20,
                 pre_train_crop='center_crop',
                 train_crop='uniform_crop',
                 valid_test_crop='center_crop'):
        self.train_image_dir = train_image_dir
        self.test_image_dir = test_image_dir
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.cache_size = (self.batch_size * cache_size_factor) # size in images
        self.center = center
        self.mean = None
        self.normalize = normalize
        self.std = None
        self.amplify = amplify
        self.train_set_flipper = ImageFlipOracle(train_flip)
        test_set_flipper = ImageFlipOracle(test_flip)
        self.train_flip_lambda = self.train_set_flipper.get_flip_lambda(train_flip)
        self.valid_flip_lambda = self.train_set_flipper.get_flip_lambda(valid_flip, deterministic=True)
        self.test_flip_lambda = test_set_flipper.get_flip_lambda(test_flip, deterministic=True)
        self.valid_dataset_size = valid_dataset_size
        self.random_seed = random_seed
        self.sample_class = sample_class
        self.custom_distribution = custom_distribution
        color_cast_oracle = ColorCastOracle(self.image_shape[-1], color_cast_range)
        self.train_color_cast_lambda = color_cast_oracle.get_color_cast_lambda(train_color_cast)
        self.valid_color_cast_lambda = color_cast_oracle.get_color_cast_lambda(valid_color_cast)
        self.test_color_cast_lambda = color_cast_oracle.get_color_cast_lambda(test_color_cast)
        crop_oracle = CropOracle(self.image_shape[0])
        self.pre_train_crop_lambda = crop_oracle.get_crop_lambda(pre_train_crop)
        self.train_crop_lambda = crop_oracle.get_crop_lambda(train_crop)
        self.valid_test_crop_lambda = crop_oracle.get_crop_lambda(valid_test_crop)

        # look for label csv 1 dir above training dir
        train_labels_csv_path = "%s/trainLabels.csv" % '/'.join(train_image_dir.split('/')[:-2])
        bd = BlockDesigner(train_labels_csv_path, seed=self.random_seed)
        self.K = bd.K

        valid_examples = bd.break_off_block(self.valid_dataset_size)
        self.train_examples = bd.remainder()
        self.n_train_batches = int(bd.size() / self.batch_size)

        self.valid_dataset = self.setup_valid_dataset(valid_examples)
        self.train_dataset = None if shuffle else self.setup_train_dataset()
        self.test_dataset = self.setup_test_dataset()
        self.n_test_examples = len(self.test_dataset["X"])

        if self.sample_class:
            self.n_train_batches = int(len(self.train_dataset["X"]) / self.batch_size) # override in case Sampler is used (TODO make this neater)
        self.train_dataset_size = self.n_train_batches * self.batch_size

        if self.center == 1 or self.normalize == 1:
            self.calc_mean_std_image()

    def valid_set(self):
        all_val_images = numpy.zeros(((len(self.valid_dataset["y"]),) + self.image_shape), dtype=theano.config.floatX)
        for i, image in enumerate(self.valid_dataset["X"]):
            all_val_images[i, ...] = self.feed_image(image, self.train_image_dir, self.valid_test_crop_lambda,
                                                     self.valid_flip_lambda, self.valid_color_cast_lambda) # b01c, Theano: bc01 CudaConvnet: c01b
        return numpy.rollaxis(all_val_images, 3, 1), numpy.array(self.valid_dataset["y"], dtype='int32')

    def train_buffer(self, new_flip_noise=None):
        """
        Yields a x_cache_block, has a size that is a multiple of training batches
        """
        if new_flip_noise:
            self.train_set_flipper.reset_noise(new_flip_noise)
        train_dataset = self.train_dataset or self.setup_train_dataset()
        x_cache_block = numpy.zeros(((self.cache_size,) + self.image_shape), dtype=theano.config.floatX)
        n_cache_blocks = int(len(train_dataset["y"]) / float(self.cache_size)) # rounding down skips the leftovers
        if not n_cache_blocks:
            raise ValueError("Train dataset length %i is too small for cache size %i" % (len(train_dataset["y"]), self.cache_size))
        for ith_cache_block in xrange(n_cache_blocks):
            ith_cache_block_end = (ith_cache_block + 1) * self.cache_size
            ith_cache_block_slice = slice(ith_cache_block * self.cache_size, ith_cache_block_end)
            for i, image in enumerate(train_dataset["X"][ith_cache_block_slice]):
                x_cache_block[i, ...] = self.feed_image(image, self.train_image_dir, self.train_crop_lambda,
                                                        self.train_flip_lambda, self.train_color_cast_lambda)
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
                x_cache_block[i, ...] = self.feed_image(image, self.test_image_dir, self.valid_test_crop_lambda,
                                                        self.test_flip_lambda, self.test_color_cast_lambda)
            yield numpy.rollaxis(x_cache_block, 3, 1), numpy.array(idxs_to_full_dataset, dtype='int32')
        # sneak the leftovers out, padded by the previous full cache block
        if n_leftovers:
            leftover_slice = slice(ith_cache_block_end, ith_cache_block_end + n_leftovers)
            for i, image in enumerate(self.test_dataset["X"][leftover_slice]):
                idxs_to_full_dataset[i] = ith_cache_block_end + i
                x_cache_block[i, ...] = self.feed_image(image, self.test_image_dir, self.valid_test_crop_lambda,
                                                        self.test_flip_lambda, self.test_color_cast_lambda)
            yield numpy.rollaxis(x_cache_block, 3, 1), numpy.array(idxs_to_full_dataset, dtype='int32')

    def read_image(self, image_name, image_dir, crop_lambda, extension=".png"):
        """
        :type image: string
        """
        as_grey = True if self.image_shape[2] == 1 else False
        img = imread(image_dir + image_name + extension, as_grey=as_grey)
        img = self.crop_image(img, crop_lambda)
        return (img.reshape(self.image_shape) / 255.) # reshape in case it is as_grey

    def preprocess_image(self, image, flip_coords, color_cast):
        """
        Important, use with read_image. This method assumes image is already
        standardized to have [0,1] pixel values
        """
        image = self.flip_image(image, flip_coords)
        image = self.color_cast_image(image, color_cast)
        if not self.mean == None:
            image = image - self.mean
        if not self.std == None:
            image = image / (self.std + 1e-5)
        return self.amplify * image

    def crop_image(self, img, crop_lambda):
        t,b,l,r = crop_lambda(img)
        assert(b-t == self.image_shape[0])
        assert(r-l == self.image_shape[1])
        return img[t:b, l:r, :]

    def color_cast_image(self, image, color_cast, masked=False):
        if masked:
            # Observed to perform much worse
            coloring = numpy.zeros(image.shape) + color_cast
            mask = (image > 0) / 255.
            return(image + (mask * coloring))
        else:
            return(image + (color_cast/255.0))

    def flip_image(self, image, flip_coords):
        assert(len(flip_coords) == 2)
        assert(max(flip_coords) <= 1)
        assert(min(flip_coords) >= 0)
        if flip_coords[0] == 1:
            image = numpy.flipud(image)
        if flip_coords[1] == 1:
            image = numpy.fliplr(image)
        return image

    def feed_image(self, image_name, image_dir, crop_lambda=None, flip_lambda=None, color_cast_lambda=None):
        img = self.read_image(image_name, image_dir, crop_lambda)
        flip_coords = flip_lambda(image_name) if flip_lambda else numpy.zeros(2)
        color_cast = color_cast_lambda() if color_cast_lambda else numpy.zeros(self.image_shape[-1])
        return self.preprocess_image(img, flip_coords, color_cast)

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
                img = self.read_image(image, self.train_image_dir, self.pre_train_crop_lambda)
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
        if self.sample_class:
            samp = Sampler(bd.remainder(), seed=self.random_seed)
            images, labels = samp.custom_distribution(self.sample_class, self.batch_size, self.custom_distribution)
            return {"X": images, "y": labels}
        else:
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
