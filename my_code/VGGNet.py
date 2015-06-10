# -*- coding: utf-8 -*-

# mainstream
import os
import sys
import time
import collections
import uuid
import json
import cPickle
import numpy
# Deep Learning
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import lasagne
from lasagne import layers
# git submodules
from ciresan.code.ciresan2012 import Ciresan2012Column
# this repo
from my_code.util import QWK, print_confusion_matrix
from my_code.data_stream import DataStream

import pdb

class VGGNet(Ciresan2012Column):
    def __init__(self, data_stream, batch_size, learning_rate, momentum, cuda_convnet, leakiness, partial_sum, model_spec, pad=1, params=None):
        self.column_params = [batch_size, cuda_convnet, leakiness, partial_sum, model_spec, pad]
        layer_input_sizes, layer_parameter_counts = self.precompute_layer_sizes(model_spec, pad, cuda_convnet)
        print "[DEBUG] all_layers input widths: {}".format(layer_input_sizes)
        print "[INFO] Estimated memory usage is %f MB per input image" % round(sum(layer_parameter_counts) * 4e-6 * 3, 2)
        # data setup
        self.ds = data_stream
        self.n_valid_batches = len(self.ds.valid_dataset) // batch_size
        self.n_train_batches = len(self.ds.train_dataset) // batch_size

        self.train_x, self.train_y = self.ds.train_buffer().next()
        self.batches_per_cache_block = len(self.train_x) // batch_size
        self.train_x = theano.shared(lasagne.utils.floatX(self.train_x))
        self.train_y = theano.shared(self.train_y)

        valid_x, self.valid_y = self.ds.valid_set()
        valid_x = theano.shared(lasagne.utils.floatX(valid_x))
        self.valid_y = theano.shared(self.valid_y)
        # build model
        all_layers = self.build_model(model_spec, layer_input_sizes, pad)

        cache_block_index = T.iscalar('cache_block_index')
        X_batch = T.tensor4('x')
        y_batch = T.imatrix('y')

        batch_slice = slice(cache_block_index * batch_size, (cache_block_index + 1) * batch_size)

        objective = lasagne.objectives.Objective(all_layers[-1], loss_function=lasagne.objectives.mse)

        loss_train = objective.get_loss(X_batch, target=y_batch)
        loss_valid = objective.get_loss(X_batch, target=y_batch, deterministic=True)

        pred = T.iround(all_layers[-1].get_output(X_batch, deterministic=True))

        self.params = lasagne.layers.get_all_params(all_layers[-1])
        updates = lasagne.updates.nesterov_momentum(loss_train, self.params, learning_rate, momentum)

        print("Compiling...")

        self.train_batch = theano.function(
            [cache_block_index], loss_train,
            updates=updates,
            givens={
                X_batch: self.train_x[batch_slice],
                y_batch: self.train_y[batch_slice],
            },
        )
        self.validate_batch = theano.function(
            [cache_block_index], [loss_valid, pred],
            givens={
                X_batch: valid_x[batch_slice],
                y_batch: self.valid_y[batch_slice],
            },
        )

    def precompute_layer_sizes(self, model_spec, pad, cuda_convnet):
        layer_input_sizes = numpy.ones(len(model_spec), dtype=int)
        layer_input_sizes[0] = model_spec[0][0]
        layer_input_sizes[1] = layer_input_sizes[0]
        for i in xrange(2,len(model_spec)):
            downsample = model_spec[i-1][2] if (len(model_spec[i-1]) >= 3) else 1
            if (len(model_spec[i-1]) >= 3) or (i == 1):
                # int division will automatically round down to match ignore_border=T
                # in theano.tensor.signal.downsample.max_pool_2d
                if pad:
                    assert(model_spec[i-1][0] - 2*pad == 1) # must be able to handle edge pixels (plus no even conv filters allowed)
                    # additive = 0 if (cuda_convnet and (len(model_spec[i]) >= 3)) else 2 # can't remember what this has to do with (maybe it is with odd sizes? will discover later)
                    additive = 0 if cuda_convnet else 2
                    layer_input_sizes[i] = (layer_input_sizes[i-1] + additive) / downsample
                else: #(prev_size - cur_conv / maxpool degree)
                    layer_input_sizes[i] = ((layer_input_sizes[i-1] - model_spec[i-1][0]) / downsample) + 1
        layer_parameter_counts = [0] + [model_spec[i - 1][1]*layer_input_sizes[i]**2 for i in xrange(1,len(model_spec))]
        return [layer_input_sizes, layer_parameter_counts]

    def build_model(self, model_spec, layer_input_sizes, pad):
        print("Building model from JSON...")
        all_layers = [layers.InputLayer(shape=(None, model_spec[0][1], model_spec[0][0], model_spec[0][0]))]
        for i in xrange(1,len(model_spec)):
            cs = model_spec[i] # current spec
            iz = layer_input_sizes[i] # input size
            ps = model_spec[i - 1] # previous spec

            prev_layer_params = ps[1] * iz**2
            fltargs = dict(n_in=prev_layer_params, n_out=cs[1]) # n_out only relevant for FC all_layers
            print "[DEBUG] Layer %i" % i
            if len(cs) >= 3: # CONV layer
                image_shape = (ps[1], iz, iz, batch_size) if cuda_convnet else (batch_size, ps[1], iz, iz)
                filter_shape = (ps[1], cs[0], cs[0], cs[1]) if cuda_convnet else (cs[1], ps[1], cs[0], cs[0])
                print image_shape
                print filter_shape
                border_mode = 'full' if pad else 'valid'
                all_layers.append(layers.Conv2DLayer(all_layers[-1],
                                    num_filters=cs[1],
                                    filter_size=(cs[0], cs[0]),
                                    W=lasagne.init.Normal()))
                all_layers.append(layers.MaxPool2DLayer(all_layers[-1], (cs[-1], cs[-1])))
            elif len(cs) == 2: # FC layer
                print fltargs
                all_layers.append((layers.DenseLayer(all_layers[-1],
                                   num_units=cs[1],
                                   W=lasagne.init.Normal())))
                all_layers.append(lasagne.layers.DropoutLayer(all_layers[-1], p=0.5))
            else:
                raise NotImplementedError()
        # output layer
        all_layers.append(layers.DenseLayer(all_layers[-1],
                           num_units=1,
                           nonlinearity=None))
        return all_layers

    def train_epoch(self):
        """
        Is responsible for moving the data stream into the column's variables
        :return: minibatch training error
        """
        for x_cache_block, y_cache_block in self.ds.train_buffer():
            self.train_x.set_value(lasagne.utils.floatX(x_cache_block), borrow=True)
            self.train_y.set_value(y_cache_block, borrow=True)

            for i in xrange(self.batches_per_cache_block):
                batch_loss = self.train_batch(i)
                yield batch_loss

    def validate(self, silent=False):
        """
        Iterates through validation minibatches
        """
        batch_valid_losses = []
        valid_predictions = []
        for j in range(self.n_valid_batches):
            batch_valid_loss, prediction = self.validate_batch(j)
            batch_valid_losses.append(batch_valid_loss)
            valid_predictions.extend(prediction)
        [kappa, M] = QWK(self.valid_y.get_value(borrow=True), numpy.array(valid_predictions))
        if not silent:
            print_confusion_matrix(M)
        val_loss = numpy.mean(batch_valid_losses)
        return [val_loss, kappa]

    def train_column(self, n_epochs):
        start_time = time.clock()
        epoch = 0
        validation_frequency = 24 # in multiples of minibatches, i.e. iters
        iter = 0
        self.historical_train_losses = []
        self.historical_val_losses = []
        self.historical_val_kappas = []
        while epoch < n_epochs:
            epoch += 1
            for batch_train_loss in self.train_epoch():
                iter += 1
                self.historical_train_losses.append([iter, batch_train_loss])

                if (iter + 1) % validation_frequency == 0:
                    print 'training @ iter = %i. Cur training error is %f %%' % (iter, 100*batch_train_loss)
                    this_valid_loss, this_kappa = self.validate()
                    self.historical_val_losses.append([iter, this_valid_loss])
                    self.historical_val_kappas.append([iter, this_kappa])
                    mins_per_epoch = self.n_train_batches*(time.clock() - start_time)/(iter*60.)
                    print('     epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, iter + 1 % self.n_train_batches, self.n_train_batches,
                           this_valid_loss * 100.))
                    print('     kappa on validation set is: %f' % this_kappa)
                    print('     averaging %f mins per epoch' % mins_per_epoch)

def save_results(filename, params):
    name = filename or 'CNN_%iParams_t%i' % (len(self.params) / 2, int(time.time()))
    print('Saving Results as "%s"...' % name)
    f = open('./results/'+name+'.pkl', 'wb')
    cPickle.dump(params, f, -1)
    f.close()

def train_drnet(network, learning_rate, momentum, n_epochs, dataset,
                 batch_size, cuda_convnet, leakiness, partial_sum, center, normalize):
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
    runid = str(uuid.uuid4())
    print("[INFO] Starting runid %s" % runid)

    with open('network_specs.json') as data_file:
        network = json.load(data_file)[network]
        netspec = network['layers']
        pad = network['pad']
        input_image_size = network['rec_input_size']
        digit_normalized_width = int(input_image_size * 0.9) # only relevant if mnist

    input_image_channels = 3
    output_classes = 1

    image_shape = (input_image_size, input_image_size, input_image_channels)
    model_spec = [[input_image_size, input_image_channels]] + netspec

    data_stream = DataStream(image_shape=image_shape)

    column = VGGNet(data_stream, batch_size, learning_rate, momentum, cuda_convnet, leakiness, partial_sum, model_spec, pad=pad)
    try:
        column.train_column(n_epochs)
    except KeyboardInterrupt:
        print "[ERROR] User terminated Training, saving results"
    column.save(runid)
    save_results(runid, [column.historical_train_losses, column.historical_val_losses, column.historical_val_kappas])

if __name__ == '__main__':
    arg_names = ['command', 'network', 'dataset', 'batch_size', 'cuda_convnet', 'center', 'normalize', 'partial_sum', 'leakiness', 'learning_rate', 'n_epochs']
    arg = dict(zip(arg_names, sys.argv))

    network = arg.get('network') or 'vgg_mini6'
    dataset = arg.get('dataset') or 'data/train_simple_crop.npz'
    batch_size = int(arg.get('batch_size') or 2)
    cuda_convnet = int(arg.get('cuda_convnet') or 0)
    center = int(arg.get('center') or 0)
    normalize = int(arg.get('normalize') or 0)
    partial_sum = int(arg.get('partial_sum') or 0) or None # 0 turns into None. None or 1 work all the time (otherwise refer to pylearn2 docs)
    leakiness = float(arg.get('leakiness') or 0.01)
    learning_rate = float(arg.get('learning_rate') or 0.01)
    momentum = float(arg.get('momentum') or 0.9)
    n_epochs = int(arg.get('n_epochs') or 800) # useful to change to 1 for a quick test run

    train_drnet(network=network, learning_rate=learning_rate, momentum=momentum, n_epochs=n_epochs, dataset=dataset, batch_size=batch_size, cuda_convnet=cuda_convnet, leakiness=leakiness, partial_sum=partial_sum, center=center, normalize=normalize)
