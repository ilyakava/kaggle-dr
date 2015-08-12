from os import listdir, path
import re
import numpy
import cPickle

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
        self.source = lasagne.utils.floatX(imread("%s.png" % test_imagepath))
        self.source_size = numpy.array(self.source.shape[:2])
        self.nn_image_size = nn_image_size

        self.octave_sizes, self.octave_tile_corners = calculate_octave_and_tile_sizes(self.source_size, self.nn_image_size,
                                                                                      max_octaves=max_octaves, octave_scale=octave_scale)
        self.batch_sizes = [len(tiles) for tiles in self.octave_tile_corners]
        # batch size will need to be kept constant (1 octave = 1 batch, processed n_itr times)
        self.max_batch_size = max(self.batch_sizes)
        print("Dreaming with batch size: %i" % self.max_batch_size)
        assert(self.max_batch_size <= 128)

    def update_source(self, batch_gradients, octave, step_size):
        gradient_tiles = numpy.rollaxis(batch_gradients, 1,4)
        # Untile the gradient
        untiled_gradient = numpy.zeros(self.octave_sizes[octave] + [3], dtype=theano.config.floatX)
        untiled_gradient_normalizer = numpy.zeros(self.octave_sizes[octave] + [3], dtype=int)

        for j, tile in enumerate(self.octave_tile_corners[octave]):
            t,l = tile
            b,r = [d+self.nn_image_size for d in tile]
            # normalize by mean per tile
            # google_lambda = step_size / abs(gradient_tiles[j]).mean()
            untiled_gradient[t:b,l:r,:] += gradient_tiles[j]
            untiled_gradient_normalizer[t:b,l:r,:] += 1
        normalized_untiled_gradient = untiled_gradient / untiled_gradient_normalizer

        # Then enlarge the gradient to source size
        zoom_in = (self.source_size / numpy.array(self.octave_sizes[octave], dtype=float)).tolist() + [1]
        source_size_gradient = nd.zoom(normalized_untiled_gradient, zoom_in, order=1)
        assert(list(source_size_gradient.shape[:2]) == self.source_size.tolist())
        # Apply the enlarged gradient to the source (clip and prevent overblow)
        old_mean = self.source.mean(axis=(0,1))
        # google_lambda = step_size / abs(source_size_gradient).mean()
        percent_lambda = (step_size*abs(self.source).max()) / abs(source_size_gradient).max()
        self.source += percent_lambda * source_size_gradient
        self.source = (self.source / self.source.mean(axis=(0,1))) * old_mean # multiply mean down
        # self.source = self.source - (self.source.mean(axis=(0,1)) - old_mean) # subtract mean down (high contrast kept)
        self.source = numpy.clip(self.source, 0.0, 255.0)

    def tile_source_into_batch(self, octave, mean=None, std=None):
        octave_size = self.octave_sizes[octave]
        tiles = self.octave_tile_corners[octave]
        zoom_out = (numpy.array(octave_size, dtype=float) / self.source_size).tolist() + [1]
        # shrink source image
        octave_image = nd.zoom(self.source, zoom_out, order=1)
        assert(list(octave_image.shape[:2]) == octave_size)

        batch = numpy.zeros((self.max_batch_size,self.nn_image_size,self.nn_image_size,3), dtype=theano.config.floatX)
        for j, tile in enumerate(tiles):
            t,l = tile
            b,r = [d+self.nn_image_size for d in tile]
            crop = octave_image[t:b,l:r,:]
            if not mean == None:
                crop = crop - mean
            if not std == None:
                crop = crop / (std + 1e-5)
            batch[j] = crop

        self.previous_batch = batch
        return numpy.rollaxis(self.previous_batch, 3, 1)

class DreamNet(VGGNet):
    def __init__(self, data_stream, batch_size, init_learning_rate, momentum,
                 leak_alpha, model_spec, loss_type, num_output_classes, pad,
                 image_shape, filter_shape, cuda_convnet=1, runid=None):
        super(DreamNet, self).__init__(data_stream, batch_size, init_learning_rate, momentum,
                                     leak_alpha, model_spec, loss_type, num_output_classes, pad,
                                     image_shape, filter_shape, cuda_convnet=1, runid=None)

        X_batch = T.tensor4('x2')
        layer_idx_of_interest = 10
        my_input = X_batch
        l2_activations = T.sum(lasagne.layers.get_output(self.all_layers[layer_idx_of_interest], my_input, deterministic=True) ** 2)
        dream_updates = lasagne.updates.sgd(l2_activations, [my_input], 1)
        self.dream_batch = theano.function(
            [],
            dream_updates.values()[0],
            givens={
                X_batch: self.x_buffer
            }
        )

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

    column = DreamNet(data_stream, batch_size, init_learning_rate, momentum, leak_alpha, model_spec, loss_type, num_output_classes, pad, image_shape, filter_shape, cuda_convnet)
    column.restore(model_file)
    return column

def plot_dreams(model_file, test_imagepath, itr_per_octave, step_size, max_octaves, octave_scale, **kwargs):
    assert(model_file)
    runid = model_runid(model_file)

    nn_image_size = get_nn_image_size(model_file)

    # precalc octaves and tiles, as well as max batch size
    dsb = DreamStudyBuffer(test_imagepath, nn_image_size, max_octaves, octave_scale)
    max_nn_pass = len(dsb.octave_sizes) * itr_per_octave

    column = load_column(model_file, batch_size=dsb.max_batch_size, **kwargs)

    try:
        nn_pass = 0
        for octave, octave_size in reversed(list(enumerate(dsb.octave_sizes))):
            for itr in range(itr_per_octave):
                next_batch = dsb.tile_source_into_batch( octave, column.ds.mean, column.ds.std )
                column.x_buffer.set_value(lasagne.utils.floatX(next_batch), borrow=True)

                batch_updates = column.dream_batch(1)

                dsb.update_source( batch_updates, octave, step_size )

                nn_pass += 1
                if (nn_pass in set([0] + [int(i) for i in numpy.logspace(0,numpy.log10(max_nn_pass),10)])):
                    name = 'data/dreams/%i_nnpass_%i_itr_%i_octave.png' % (nn_pass, itr, octave)
                    print("saving %s" % name)
                    scipy.misc.toimage(numpy.uint8(dsb.source)).save(name, "PNG")

    except KeyboardInterrupt:
        print "[ERROR] User terminated Dream Study"
    print "Done"

if __name__ == '__main__':
    _ = args.get()

    plot_dreams(model_file=_.model_file,
           test_imagepath=_.test_imagepath,
           itr_per_octave=_.itr_per_octave,
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
