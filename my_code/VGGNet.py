# -*- coding: utf-8 -*-

import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from lasagne.nonlinearities import LeakyRectify

from ciresan.code.logistic_sgd import LogisticRegression, load_data
from ciresan.code.mlp import HiddenLayer
from ciresan.code.ciresan2012 import save_model, Ciresan2012Column
from ciresan.code.convolutional_mlp import LeNetConvPoolLayer

import cPickle
import collections

import pdb

class NetworkInput(object):
    """
    A dummy class that pretends the input to the network is a 'layer'
    """
    def __init__(self, input):
        self.output = input

class VGGNet(Ciresan2012Column):
    def __init__(self, datasets, batch_size=128, cuda_convnet=0, leakiness=0.01, params=None):
        """
        :param datasets: Array of train, val, test x,y tuples

        :param params: W/b weights in the order ...
        """
        rng = numpy.random.RandomState(23455)

        # TODO: could make this a theano sym variable to abstract
        # loaded data from column instantiation
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        # TODO: could move this to train method
        # compute number of minibatches for training, validation and testing
        self.n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        self.n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        self.n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        self.n_train_batches /= batch_size
        self.n_valid_batches /= batch_size
        self.n_test_batches /= batch_size

        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        learning_rate = T.fscalar()

        # start-snippet-1
        x = T.matrix('x')   # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels
        lr = LeakyRectify(leakiness)

        # TODO make this an argument to the class
        vggneta = [
            (3,64,2), # 64 3x3 filters with 2x2 maxpooling after
            (3,128,2),
            (3,256,1), # no maxpool after
            (3,256,2),
            (3,512,1),
            (3,512,2),
            (3,512,1),
            (3,512,2),
            (1, 4096), # FC layer
            (1, 4096),
            (1, 5), # FC, probably thin this out a different way
            (1, 5) # softmax
        ]

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the column'

        # EDIT ME
        model_spec = [(224, 1)] + vggneta # input with 3 channels and size 224
        # EDIT ME

        # list to hold layers, each element must have an `output` property,
        # each element after the first one must also have a `params` property
        layers = [None] * len(model_spec)
        # stick in the images as "0th" layer
        # TODO get rid of dimshuffle
        raw_image_data = x.reshape((batch_size, model_spec[0][1], model_spec[0][0], model_spec[0][0]))
        network_input = raw_image_data.dimshuffle(1, 2, 3, 0) if cuda_convnet else raw_image_data
        layers[0] = NetworkInput(raw_image_data)

        # precompute layer sizes: not (prev_size - cur_conv / maxpool degree) since we are padding images
        layer_sizes_ignore_pool = numpy.ones(len(model_spec), dtype=int)
        layer_sizes = numpy.ones(len(model_spec), dtype=int)
        layer_sizes_ignore_pool[0] = model_spec[0][0] # input image size
        layer_sizes[0] = model_spec[0][0]
        for i in xrange(1,len(model_spec)):
            if len(model_spec[i]) == 3:
                layer_sizes_ignore_pool[i] = layer_sizes_ignore_pool[i-1] # size stays the same with padding
                # will automatically round down to match ignore_border=T in theano.tensor.signal.downsample.max_pool_2d
                layer_sizes[i] = layer_sizes_ignore_pool[i-1] / model_spec[i][2]

        print layer_sizes
        # create layers
        for i in xrange(1,len(model_spec)):
            cs = model_spec[i] # current spec
            cz = layer_sizes_ignore_pool[i] # current size (given to conv op, so pool needs to be ignored)
            ps = model_spec[i - 1] # previous spec
            pz = layer_sizes[i - 1] # previous size (as will be input into current layer, i.e. take into account pool)

            fltargs = dict(n_in=ps[1] * pz**2, n_out=cs[1])
            print i
            if len(cs) == 3: # conv layer
                image_shape = (ps[1], cz, cz, batch_size) if cuda_convnet else (batch_size, ps[1], cz, cz)
                filter_shape = (ps[1], cs[0], cs[0], cs[1]) if cuda_convnet else (cs[1], ps[1], cs[0], cs[0])
                print image_shape
                print filter_shape
                layers[i] = LeNetConvPoolLayer(
                    rng,
                    input=layers[i-1].output,
                    filter_shape=filter_shape, # (prev_thickness, convx, convy, cur_thickness)
                    image_shape=image_shape, # (prev_thickness, cur_size, cur_size, bs)
                    poolsize=(cs[2], cs[2]),
                    cuda_convnet=cuda_convnet,
                    activation=lr,
                    border_mode='full'
                )
            elif i == (len(model_spec) - 1): # last softmax layer
                assert(len(ps) == 2) # must follow an FC layer
                layers[i] = LogisticRegression(input=layers[i-1].output, **fltargs)
            elif len(cs) == 2: # FC layer
                raw_in = layers[i-1].output
                if len(ps) == 3: # previous layer was a conv layer
                    flt_input = raw_in.dimshuffle(3, 0, 1, 2).flatten(2) if cuda_convnet else raw_in.flatten(2)
                else:
                    flt_input = raw_in
                layers[i] = HiddenLayer(rng, input=flt_input, activation=T.tanh, **fltargs)

        # TODO change this to the kappa loss
        cost = layers[-1].negative_log_likelihood(y)

        # create a function to compute the mistakes that are made by the model
        self.test_model = theano.function(
            [index],
            layers[-1].errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        # create a function to compute probabilities of all output classes
        self.test_output_batch = theano.function(
            [index],
            layers[-1].p_y_given_x,
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size]
            }
        )

        self.validate_model = theano.function(
            [index],
            layers[-1].errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        # create a list of all model parameters to be fit by gradient descent
        nonflat_params = [layer.params for layer in layers[1:]]
        self.params = [item for sublist in nonflat_params for item in sublist]
        self.column_params = [model_spec, batch_size, cuda_convnet]

        # create a list of gradients for all model parameters
        grads = T.grad(cost, self.params)

        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        updates = [
            (param_i, param_i - (learning_rate) * grad_i)
            for param_i, grad_i in zip(self.params, grads)
        ]

        # Suggested by Alex Krizhevsky, found on:
        # http://yyue.blogspot.com/2015/01/a-brief-overview-of-deep-learning.html
        optimal_ratio = 0.001
        # should show what multiple current learning rate is of optimal learning rate
        grads_L1 = sum([abs(grad).sum() for grad in grads])
        params_L1 = sum([abs(param).sum() for param in self.params])
        update_ratio = (learning_rate/(optimal_ratio)) * (grads_L1/params_L1)

        self.train_model = theano.function(
            [index, learning_rate],
            [cost, update_ratio],
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

def train_vggnet(init_learning_rate=0.001, n_epochs=800,
                 dataset='mnist.pkl.gz', batch_size=1000, cuda_convnet=0, leakiness=0.01):
    """
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """
    datasets = load_data(dataset, 0, 224)
    column = VGGNet(datasets, batch_size, cuda_convnet, leakiness)
    column.train_column(init_learning_rate, n_epochs)


if __name__ == '__main__':
    arg_names = ['command', 'batch_size', 'cuda_convnet', 'leakiness', 'init_learning_rate', 'n_epochs']
    arg = dict(zip(arg_names, sys.argv))

    batch_size = int(arg.get('batch_size') or 2)
    cuda_convnet = int(arg.get('cuda_convnet') or 0)
    leakiness = int(arg.get('leakiness') or 0.01)
    init_learning_rate = float(arg.get('init_learning_rate') or 0.001)
    n_epochs = int(arg.get('n_epochs') or 800) # useful to change to 1 for a quick test run

    train_vggnet(init_learning_rate=init_learning_rate, n_epochs=n_epochs, batch_size=batch_size, cuda_convnet=cuda_convnet)
