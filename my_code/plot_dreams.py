from os import listdir, path
import re
import numpy
import cPickle

# import matplotlib
# matplotlib.use('Agg')
# from skimage.io import imread
# matplotlib.rcParams.update({'font.size': 2})
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import AxesGrid

import theano
import lasagne

from my_code.VGGNet import VGGNet
from my_code.data_stream import DataStream

import my_code.dream_args as args
from my_code.predict import model_runid

import scipy.misc
import scipy.ndimage as nd
from skimage.io import imread
from PIL import Image

import pdb

def calculate_octave_and_tile_sizes(source_size, nn_image_size, max_octaves=4, octave_scale=1.4, overlap_percentage=0.25):
    """
    :type source_size: Array of 2 integers
    :param source_size: [height, width] of image to have the dream

    :type nn_image_size: integer
    """
    # find octave sizes
    # array of [h,w] arrays
    octave_sizes = [list(source_size)]
    while len(octave_sizes) < max_octaves and min(octave_sizes[-1]) > nn_image_size:
        min_dim = min(octave_sizes[-1])
        scale = min(octave_scale, float(min_dim) / nn_image_size)

        new_dims = [int(dim / scale) for dim in octave_sizes[-1]]
        octave_sizes.append(new_dims)
    assert(numpy.array(octave_sizes).min() >= nn_image_size)

    # calculate tile limits per octave (and normalizing coefs)
    octave_tile_corners = []
    for size in octave_sizes:
        h,w = size
        max_h = (h-nn_image_size); max_w = (w-nn_image_size);
        stride = int(nn_image_size - overlap_percentage*nn_image_size)

        tops = [0]
        while tops[-1] < max_h:
            tops.append(tops[-1]+stride)
        tops[-1] = max_h

        lefts = [0]
        while lefts[-1] < max_w:
            lefts.append(lefts[-1]+stride)
        lefts[-1] = max_w

        tile_corners = []
        for top in tops:
            for left in lefts:
                if not [top,left] in tile_corners:
                    tile_corners.append([top,left])
        octave_tile_corners.append(tile_corners)
    return(octave_sizes,octave_tile_corners)

class DreamStudyBuffer(object):
    """
    Keeps the state of the dream in a double buffer
    (source->batch, batch_output->source, and repeat)
    """

    def __init__(self, test_imagepath, nn_image_size, max_octaves, octave_scale):
        """
        :type source_size: Array of 2 integers
        :param source_size: [height, width] of image to have the dream

        :type nn_image_size: integer
        """
        self.source = imread("%s.png" % test_imagepath)
        self.source_size = numpy.array(self.source.shape[:2])
        self.nn_image_size = nn_image_size

        self.octave_sizes, self.octave_tile_corners = calculate_octave_and_tile_sizes(self.source_size, self.nn_image_size,
                                                                                      max_octaves=max_octaves, octave_scale=octave_scale)
        self.batch_size = sum([len(tiles) for tiles in self.octave_tile_corners])
        assert(self.batch_size <= 128)

    def update_source(self, batch_gradients, step_size=0.5):
        batch_gradients = numpy.rollaxis(batch_gradients, 1,4)
        batch_images = self.previous_batch

        octave_images = [numpy.zeros(size + [3], dtype=theano.config.floatX) for size in self.octave_sizes]
        octave_accs = [numpy.zeros(size + [3], dtype=int) for size in self.octave_sizes]
        idx = 0
        for i, tiles in enumerate(self.octave_tile_corners):
            octave_image = octave_images[i]
            octave_acc = octave_accs[i]
            for j, tile in enumerate(tiles):
                t,l = tile
                b,r = [d+self.nn_image_size for d in tile]

                mean_lambda = (step_size*abs(batch_images[idx]).mean()) / abs(batch_gradients[idx]).mean()
                # google_lambda = (step_size) / abs(batch_gradients[idx]).mean()
                new_image = batch_images[idx] + (mean_lambda * batch_gradients[idx])
                octave_image[t:b,l:r,:] += new_image

                octave_acc[t:b,l:r,:] += 1
                idx += 1
        # now average together the pixels in the many images
        normalized_octave_images = [octave_images[i] / (len(self.octave_sizes)*octave_accs[i]) for i in range(len(octave_images))]
        # enlarge each octave to original image size and update source image
        cumulative_image = normalized_octave_images[0]
        for normalized_octave_image in normalized_octave_images[1:]:
            new_zoom = (self.source_size / numpy.array(normalized_octave_image.shape[:2], dtype=float)).tolist() + [1]
            enlarged = nd.zoom(normalized_octave_image, new_zoom, order=1)
            assert(list(enlarged.shape[:2]) == self.source_size.tolist())
            cumulative_image += enlarged

        self.source = cumulative_image

    def serve_batch(self, mean=None, std=None):
        # skip resizing source
        octave_images = [self.source]
        for new_size in self.octave_sizes[1:]:
            new_zoom = (numpy.array(new_size, dtype=float) / self.source_size).tolist() + [1]
            shrunken = nd.zoom(self.source, new_zoom, order=1)
            assert(list(shrunken.shape[:2]) == new_size)
            octave_images.append(shrunken)

        batch = numpy.zeros((self.batch_size,self.nn_image_size,self.nn_image_size,3), dtype=theano.config.floatX)
        idx = 0
        for i, tiles in enumerate(self.octave_tile_corners):
            octave_image = octave_images[i]
            for j, tile in enumerate(tiles):
                t,l = tile
                b,r = [d+self.nn_image_size for d in tile]
                crop = octave_image[t:b,l:r,:]
                if not mean == None:
                    crop = crop - mean
                if not std == None:
                    crop = crop / (std + 1e-5)
                batch[idx] = crop
                idx += 1

        self.previous_batch = batch
        return numpy.rollaxis(self.previous_batch, 3, 1)

# Layers to choose:

# 1: ShuffleBC01ToC01BLayer
# 2: Conv2DCCLayer
# 3: MaxPool2DCCLayer
# 4: DropoutLayer
# 5: Conv2DCCLayer
# 6: MaxPool2DCCLayer
# 7: DropoutLayer
# 8: Conv2DCCLayer
# 9: DropoutLayer
# 10: Conv2DCCLayer
# 11: MaxPool2DCCLayer
# 12: DropoutLayer
# 13: Conv2DCCLayer
# 14: MaxPool2DCCLayer
# 15: DropoutLayer
# 16: Conv2DCCLayer
# 17: MaxPool2DCCLayer
# 18: ShuffleC01BToBC01Layer
# 19: DropoutLayer
# 20: DenseLayer
# 21: FeaturePoolLayer
# 22: DropoutLayer
# 23: DenseLayer
# 24: FeaturePoolLayer
# 25: DropoutLayer
# 26: DenseLayer

def get_nn_image_size(model_file):
    f = open(model_file)
    _batch_size, init_learning_rate, momentum, leak_alpha, model_spec, loss_type, num_output_classes, pad, image_shape = cPickle.load(f)
    f.close()
    return image_shape[0]

def load_column(model_file, batch_size, train_dataset, train_labels_csv_path, center, normalize, train_flip,
                test_dataset, random_seed, valid_dataset_size, filter_shape, cuda_convnet):
    print("Loading Model...")
    f = open(model_file)
    _batch_size, init_learning_rate, momentum, leak_alpha, model_spec, loss_type, num_output_classes, pad, image_shape = cPickle.load(f)
    data_stream = DataStream(train_image_dir=train_dataset, train_labels_csv_path=train_labels_csv_path, image_shape=image_shape, batch_size=batch_size, cache_size_factor=1, center=center, normalize=normalize, train_flip=train_flip, test_image_dir=test_dataset, random_seed=random_seed, valid_dataset_size=valid_dataset_size)
    f.close()

    column = VGGNet(data_stream, batch_size, init_learning_rate, momentum, leak_alpha, model_spec, loss_type, num_output_classes, pad, image_shape, filter_shape, cuda_convnet)
    column.restore(model_file)
    return column

def plot_dreams(model_file, test_imagepath, max_itr, step_size, max_octaves, octave_scale, **kwargs):
    assert(model_file)
    runid = model_runid(model_file)

    nn_image_size = get_nn_image_size(model_file)

    dsb = DreamStudyBuffer(test_imagepath, nn_image_size, max_octaves, octave_scale)

    column = load_column(model_file, batch_size=dsb.batch_size, **kwargs)

    try:
        itr = 0
        column.x_buffer.set_value(lasagne.utils.floatX(dsb.serve_batch(column.ds.mean, column.ds.std)), borrow=True)
        while itr <= max_itr:

            if (itr in set([0] + [int(i) for i in numpy.logspace(0,numpy.log10(max_itr),10)])):
                name = 'data/dreams/%i_itr.png' % itr
                print("saving %s" % name)
                scipy.misc.toimage(dsb.source).save(name, "PNG")

            batch_updates = column.dream_batch(1)
            dsb.update_source(batch_updates, step_size)
            column.x_buffer.set_value(lasagne.utils.floatX(dsb.serve_batch()), borrow=True)

            itr += 1

    except KeyboardInterrupt:
        print "[ERROR] User terminated Dream Study"
    print "Done"

if __name__ == '__main__':
    _ = args.get()

    plot_dreams(model_file=_.model_file,
           test_imagepath=_.test_imagepath,
           max_itr=_.max_itr,
           step_size=_.step_size,
           max_octaves=_.max_octaves,
           octave_scale=_.octave_scale,
           train_dataset=_.train_dataset,
           train_labels_csv_path=_.train_labels_csv_path,
           center=_.center,
           normalize=_.normalize,
           train_flip=_.train_flip,
           test_dataset=None,
           random_seed=_.random_seed,
           valid_dataset_size=_.valid_dataset_size,
           filter_shape=_.filter_shape,
           cuda_convnet=_.cuda_convnet)
