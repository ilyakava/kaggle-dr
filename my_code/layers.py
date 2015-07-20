import numpy
import theano
import theano.tensor as T
import lasagne

import pdb

class Fold4xChannelsLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        n_folds = 2
        midpoint, remainder = divmod(input.shape[-1], n_folds)
        tile1 = input[:, :, :midpoint, :midpoint] # tl
        tile2 = input[:, :, :midpoint, midpoint:] # tr
        tile3 = input[:, :, midpoint:, :midpoint] # bl
        tile4 = input[:, :, midpoint:, midpoint:] # br
        return T.concatenate([tile1, tile2, tile3, tile4], axis=1)

    def get_output_shape_for(self, input_shape):
        n_folds = 2
        return (input_shape[0], input_shape[1] * n_folds**2, input_shape[2] / n_folds, input_shape[3] / n_folds)

class Fold4xBatchesLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        midpoint, remainder = divmod(input.shape[-1], 2)
        tile1 = input[:, :, :midpoint, :midpoint] # 0 degrees
        tile2 = input[:, :, :midpoint, :-(midpoint+1):-1].dimshuffle(0, 1, 3, 2) # 90 degrees
        tile3 = input[:, :, :-(midpoint+1):-1, :-(midpoint+1):-1] # 180 degrees
        tile4 = input[:, :, :-(midpoint+1):-1, :midpoint].dimshuffle(0, 1, 3, 2) # 270 degrees
        return T.concatenate([tile1, tile2, tile3, tile4], axis=0)

    def get_output_shape_for(self, input_shape):
        n_folds = 2
        return (input_shape[0] * 4, input_shape[1], input_shape[2] / n_folds, input_shape[3] / n_folds)

class Unfold4xBatchesLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        input_r = input.reshape((4, self.input_shape[0] // 4, int(numpy.prod(self.input_shape[1:]))))
        return input_r.transpose(1, 0, 2).reshape(self.get_output_shape())

    def get_output_shape_for(self, input_shape):
        return input_shape[0] / 4, numpy.prod(input_shape[1:]) * 4
