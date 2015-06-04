# -*- coding: utf-8 -*-

# mainstream
import os
import sys
import time
import cPickle
import collections
import numpy
# Deep Learning
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from lasagne.nonlinearities import LeakyRectify
# git submodules
from ciresan.code.logistic_sgd import LogisticRegression, load_data
from ciresan.code.mlp import HiddenLayer
from ciresan.code.ciresan2012 import save_model, Ciresan2012Column
from ciresan.code.convolutional_mlp import LeNetConvPoolLayer
# this repo
from my_code.graphics import SKULL
import my_code.network_specs as S

import pdb

class NetworkInput(object):
    """
    A dummy class that pretends the input to the network is a 'layer'
    """
    def __init__(self, input):
        self.output = input

class VGGNet(Ciresan2012Column):
    def __init__(self, datasets, batch_size, cuda_convnet, leakiness, partial_sum, model_spec, pad=1, params=None):
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

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the column'
        # NOTE: the word "layer" in the comments and variables refers to an object
        # that can be instantiated, i.e. "layer" is an operation in the CN network
        # (with inputs and outputs) and therefore refers to a region of the network that
        # has some inputs and has some outputs. Again, "layer" does not refer to a group
        # of parameters or a block of weights. Rather, "layer" is an object that can be
        # instantiated and has params/weights as its attributes, as well as input/output
        # as attributes

        # list to hold layers, each element must have an `output` property,
        # each element after the first one must also have a `params` property
        layers = [None] * len(model_spec)
        # stick in the images as "0th" layer
        # TODO get rid of dimshuffle
        raw_image_data = x.reshape((batch_size, model_spec[0][1], model_spec[0][0], model_spec[0][0]))
        network_input = raw_image_data.dimshuffle(1, 2, 3, 0) if cuda_convnet else raw_image_data
        layers[0] = NetworkInput(network_input)

        # precompute layer sizes: not (prev_size - cur_conv / maxpool degree) since we are padding images
        layer_input_sizes = numpy.ones(len(model_spec), dtype=int)
        layer_input_sizes[0] = model_spec[0][0]
        layer_input_sizes[1] = layer_input_sizes[0]
        for i in xrange(2,len(model_spec)):
            downsample = model_spec[i-1][2] if (len(model_spec[i-1]) == 3) else 1
            if (len(model_spec[i-1]) == 3) or (i == 1):
                # int division will automatically round down to match ignore_border=T
                # in theano.tensor.signal.downsample.max_pool_2d
                if pad:
                    assert(model_spec[i-1][0] - 2*pad == 1) # must be able to handle edge pixels (plus no even conv filters allowed)
                    additive = 0 if (cuda_convnet and (len(model_spec[i]) == 3)) else 2
                    layer_input_sizes[i] = (layer_input_sizes[i-1] + additive) / downsample
                else:
                    layer_input_sizes[i] = ((layer_input_sizes[i-1] - model_spec[i-1][0]) / downsample) + 1

        # print some info
        print "[DEBUG] layers input widths: {}".format(layer_input_sizes)
        layer_parameter_counts = [0] + [model_spec[i - 1][1]*layer_input_sizes[i]**2 for i in xrange(1,len(model_spec))]
        print "[DEBUG] layer parameter sizes: {}".format(layer_parameter_counts)

        layer_weight_count = numpy.zeros(len(model_spec))
        for i in xrange(1,len(model_spec)):
            layer_weight_count[i] = model_spec[i - 1][1]*layer_input_sizes[i]**2
            if len(model_spec[i]) == 2:
                layer_weight_count[i] *= model_spec[i][1]
            else:
                layer_weight_count[i] *= batch_size
        print "[DEBUG] layer weight counts: {}".format(layer_weight_count)
        # warning when FC layers are too big
        FC_layers = numpy.array([-1] + [len(layer_spec) for layer_spec in model_spec[1:]]) == 2 # -1 to skip input
        FC_layers_descending = layer_weight_count[FC_layers].tolist() == sorted(layer_weight_count[FC_layers])[::-1]
        if not FC_layers_descending:
            print SKULL
            print "[WARNING] FC layers are not descending in weight counts"
        # info for weight increase between conv and FC layers
        first_FC_layer_index = min(numpy.where(FC_layers)[0])
        weight_increase_into_FC = layer_weight_count[first_FC_layer_index] / layer_weight_count[first_FC_layer_index-1]
        print "[INFO] Weights increased %f X from last CONV into first FC layer" % weight_increase_into_FC
        # upto 50ish probably ok (network on: http://cs231n.github.io/convolutional-networks/)

        # create layers
        for i in xrange(1,len(model_spec)):
            cs = model_spec[i] # current spec
            iz = layer_input_sizes[i] # input size
            ps = model_spec[i - 1] # previous spec

            prev_layer_params = ps[1] * iz**2
            fltargs = dict(n_in=prev_layer_params, n_out=cs[1]) # n_out only relevant for FC layers
            print "[DEBUG] %i" % i
            if len(cs) == 3: # conv layer
                image_shape = (ps[1], iz, iz, batch_size) if cuda_convnet else (batch_size, ps[1], iz, iz)
                filter_shape = (ps[1], cs[0], cs[0], cs[1]) if cuda_convnet else (cs[1], ps[1], cs[0], cs[0])
                print image_shape
                print filter_shape
                border_mode = 'full' if pad else 'valid'
                layers[i] = LeNetConvPoolLayer(
                    rng,
                    input=layers[i-1].output,
                    filter_shape=filter_shape, # (prev_thickness, convx, convy, cur_thickness)
                    image_shape=image_shape, # (prev_thickness, cur_size, cur_size, bs)
                    poolsize=(cs[2], cs[2]),
                    cuda_convnet=cuda_convnet,
                    activation=lr,
                    border_mode=border_mode,
                    partial_sum=partial_sum,
                    pad=pad
                )
            elif i == (len(model_spec) - 1): # last softmax layer
                print fltargs
                assert(len(ps) == 2) # must follow an FC layer
                layers[i] = LogisticRegression(input=layers[i-1].output, **fltargs)
            elif len(cs) == 2: # FC layer
                print fltargs
                raw_in = layers[i-1].output
                if len(ps) == 3: # previous layer was a conv layer
                    flt_input = raw_in.dimshuffle(3, 0, 1, 2).flatten(2) if cuda_convnet else raw_in.flatten(2)
                else:
                    flt_input = raw_in
                layers[i] = HiddenLayer(rng, input=flt_input, activation=T.tanh, **fltargs)

        # going off of: http://cs231n.github.io/convolutional-networks/
        print "[INFO] Estimated memory usage is %f MB per input image" % round(sum(layer_parameter_counts) * 4e-6 * 3, 2)
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
        # TODO make ciresan not depend on the following order
        # nkerns, batch_size, normalized_width, distortion, cuda_convnet
        self.column_params = [model_spec, batch_size, 0, 0, cuda_convnet]

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

def train_vggnet(init_learning_rate, n_epochs,
                 dataset, batch_size, cuda_convnet, leakiness, partial_sum, normalization):
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
    input_image_size, input_image_channels, normalized_width, pad, netspec = 69, 1, 60, 0, S.galaxynet_lessFC

    image_shape = (input_image_size, input_image_size, input_image_channels)
    model_spec = [(input_image_size, input_image_channels)] + netspec

    datasets = load_data(dataset, out_image_size=input_image_size, normalized_width=normalized_width, conserve_gpu_memory=True, center=normalization, image_shape=image_shape)

    column = VGGNet(datasets, batch_size, cuda_convnet, leakiness, partial_sum, model_spec, pad=pad)
    column.train_column(init_learning_rate, n_epochs)


if __name__ == '__main__':
    arg_names = ['command', 'dataset', 'batch_size', 'cuda_convnet', 'normalization', 'partial_sum', 'leakiness', 'init_learning_rate', 'n_epochs']
    arg = dict(zip(arg_names, sys.argv))

    dataset = arg.get('dataset') or 'data/train_simple_crop.npz'
    batch_size = int(arg.get('batch_size') or 2)
    cuda_convnet = int(arg.get('cuda_convnet') or 0)
    normalization = int(arg.get('normalization') or 0)
    partial_sum = int(arg.get('partial_sum') or 0) or None # 0 turns into None. None or 1 work all the time (otherwise refer to pylearn2 docs)
    leakiness = float(arg.get('leakiness') or 0.01)
    init_learning_rate = float(arg.get('init_learning_rate') or 0.001)
    n_epochs = int(arg.get('n_epochs') or 800) # useful to change to 1 for a quick test run

    train_vggnet(init_learning_rate=init_learning_rate, n_epochs=n_epochs, dataset=dataset, batch_size=batch_size, cuda_convnet=cuda_convnet, leakiness=leakiness, partial_sum=partial_sum, normalization=normalization)
