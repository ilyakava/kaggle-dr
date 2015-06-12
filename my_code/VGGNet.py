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
from lasagne import layers, nonlinearities
import lasagne.layers.cuda_convnet
from lasagne.nonlinearities import LeakyRectify
# git submodules
from ciresan.code.ciresan2012 import Ciresan2012Column
# this repo
from my_code.util import QWK, print_confusion_matrix, UnsupportedPredictedClasses
from my_code.data_stream import DataStream

import pdb

class VGGNet(Ciresan2012Column):
    def __init__(self, data_stream, batch_size, learning_rate, momentum, leakiness, model_spec, num_output_classes, pad=1, params=None):
        self.column_params = [batch_size, learning_rate, momentum, leakiness, model_spec, pad]
        layer_input_sizes, layer_parameter_counts = self.precompute_layer_sizes(model_spec, pad)
        print "[DEBUG] all_layers input widths: {}".format(layer_input_sizes)
        print "[INFO] Estimated memory usage is %f MB per input image" % round(sum(layer_parameter_counts) * 4e-6 * 3, 2)
        # data setup
        self.ds = data_stream
        self.n_train_batches = len(self.ds.train_dataset) // batch_size
        self.n_valid_batches = len(self.ds.valid_dataset) // batch_size
        self.batches_per_cache_block = self.ds.cache_size // batch_size
        self.num_output_classes = num_output_classes

        self.train_x, self.train_y = self.ds.train_buffer().next()
        self.train_x = theano.shared(lasagne.utils.floatX(self.train_x))
        self.train_y = T.cast(theano.shared(self.train_y), 'int32')

        valid_x, self.valid_y = self.ds.valid_set()
        valid_x = theano.shared(lasagne.utils.floatX(valid_x))
        self.valid_y = T.cast(theano.shared(self.valid_y), 'int32')

        all_layers = self.build_model(model_spec, leakiness, pad)

        cache_block_index = T.iscalar('cache_block_index')
        X_batch = T.tensor4('x')
        y_batch = T.ivector('y')
        batch_slice = slice(cache_block_index * batch_size,
                            (cache_block_index + 1) * batch_size)

        objective, pred = self.build_objective_predictions(X_batch, all_layers[-1])
        loss_train = objective.get_loss(X_batch, target=y_batch)
        loss_valid = objective.get_loss(X_batch, target=y_batch, deterministic=True)

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

    def precompute_layer_sizes(self, model_spec, pad):
        layer_input_sizes = numpy.ones(len(model_spec), dtype=int)
        layer_input_sizes[0] = model_spec[0]["size"]
        layer_input_sizes[1] = layer_input_sizes[0]
        for i in xrange(2,len(model_spec)):
            downsample = (model_spec[i-1].get("pool_stride") or 1) if (model_spec[i-1]["type"] == "CONV") else 1
            if (model_spec[i-1]["type"] == "CONV") or (i == 1):
                # int division will automatically round down to match ignore_border=T
                # in theano.tensor.signal.downsample.max_pool_2d
                if pad:
                    assert(model_spec[i-1]["filter_size"] - 2*pad == 1) # must be able to handle edge pixels (plus no even conv filters allowed)
                    # additive = 0 if (cuda_convnet and (len(model_spec[i]) >= 3)) else 2 # can't remember what this has to do with (maybe it is with odd sizes? will discover later)
                    additive = 0 if cuda_convnet else 2
                    layer_input_sizes[i] = (layer_input_sizes[i-1] + additive) / downsample
                else: #(prev_size - cur_conv / maxpool degree)
                    layer_input_sizes[i] = ((layer_input_sizes[i-1] - model_spec[i-1]["filter_size"]) / downsample) + 1
        width = model_spec[i - 1]["num_filters"] if model_spec[i-1]["type"] == "CONV" else model_spec[i - 1]["num_units"]
        layer_parameter_counts = [0] + [width*layer_input_sizes[i]**2 for i in xrange(1,len(model_spec))]
        return [layer_input_sizes, layer_parameter_counts]

    def build_objective_predictions(self, X, output):
        if self.num_output_classes > 1:
            objective = lasagne.objectives.Objective(output, loss_function=lasagne.objectives.categorical_crossentropy)
            pred = T.argmax(lasagne.layers.get_output(output, X, deterministic=True), axis=1)
        else:
            raise ValueError("unsupported output shape %i" % self.num_output_classes)
        return objective, pred

    def build_model(self, model_spec, leakiness, pad):
        print("Building model from JSON...")
        lr = LeakyRectify(leakiness)
        def get_nonlinearity(layer):
            default_nonlinear = "ReLU"  # for all Conv2DLayer, Conv2DCCLayer, and DenseLayer
            req = layer.get("nonlinearity") or default_nonlinear
            return {
                "LReLU": lr,
                "None": None,
                "sigmoid": nonlinearities.sigmoid,
                "ReLU": nonlinearities.rectify,
                "softmax": nonlinearities.softmax,
                "tanh": nonlinearities.tanh
            }[req]
        def get_init(layer):
            default_init = "GlorotUniform" # for both Conv2DLayer and DenseLayer (Conv2DCCLayer is None)
            req = layer.get("init") or default_init
            return {
                "Normal": lasagne.init.Normal(),
                "Orthogonal": lasagne.init.Orthogonal(gain='relu'),
                "GlorotUniform": lasagne.init.GlorotUniform()
            }[req]

        all_layers = [layers.InputLayer(shape=(None, model_spec[0]["channels"], model_spec[0]["size"], model_spec[0]["size"]))]
        for i in xrange(1,len(model_spec)):
            cs = model_spec[i] # current spec
            if cs["type"] == "CONV":
                border_mode = 'full' if pad else 'valid'
                if cs.get("dropout"):
                    all_layers.append(lasagne.layers.DropoutLayer(all_layers[-1], p=cs["dropout"]))
                all_layers.append(layers.cuda_convnet.Conv2DCCLayer(all_layers[-1],
                                    num_filters=cs["num_filters"],
                                    filter_size=(cs["filter_size"], cs["filter_size"]),
                                    border_mode=border_mode,
                                    W=get_init(cs),
                                    nonlinearity=get_nonlinearity(cs)))
                if cs.get("pool_size"):
                    all_layers.append(layers.cuda_convnet.MaxPool2DCCLayer(all_layers[-1],
                                        pool_size=(cs["pool_size"], cs["pool_size"]),
                                        stride=(cs["pool_stride"], cs["pool_stride"])))
            elif cs["type"] == "FC":
                if cs.get("dropout"):
                    all_layers.append(lasagne.layers.DropoutLayer(all_layers[-1], p=cs["dropout"]))
                all_layers.append(layers.DenseLayer(all_layers[-1],
                                   num_units=cs["num_units"],
                                   W=get_init(cs),
                                   nonlinearity=get_nonlinearity(cs)))
                if cs.get("pool_size"):
                    all_layers.append(layers.FeaturePoolLayer(all_layers[-1], cs["pool_size"]))
            elif cs["type"] == "OUTPUT":
                if cs.get("dropout"):
                    all_layers.append(lasagne.layers.DropoutLayer(all_layers[-1], p=cs["dropout"]))
                all_layers.append(layers.DenseLayer(all_layers[-1],
                                   num_units=self.num_output_classes,
                                   W=get_init(cs),
                                   nonlinearity=get_nonlinearity(cs)))
            else:
                raise NotImplementedError()
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
        [kappa, M] = QWK(self.valid_y.get_value(borrow=True), numpy.array(valid_predictions), self.num_output_classes)
        if not silent and (self.num_output_classes == 5):
            print_confusion_matrix(M)
        val_loss = numpy.mean(batch_valid_losses)
        return [val_loss, kappa]

    def train_column(self, n_epochs):
        validations_per_epoch = 1
        print("Training...")
        start_time = time.clock()
        epoch = 0
        batch_multiple_to_validate = self.n_train_batches // validations_per_epoch
        iter = 0
        self.historical_train_losses = []
        self.historical_val_losses = []
        self.historical_val_kappas = []
        while epoch < n_epochs:
            epoch += 1
            for batch_train_loss in self.train_epoch():
                iter += 1
                self.historical_train_losses.append([iter, batch_train_loss])

                if (iter + 1) % batch_multiple_to_validate == 0:
                    mins_per_epoch = self.n_train_batches*(time.clock() - start_time)/(iter*60.)
                    print('training @ iter = %i @ %.1fm (ETA %.1fm). Cur training error is %f %%' %
                        (iter, ((time.clock() - start_time )/60.), mins_per_epoch*(n_epochs-epoch-1), 100*batch_train_loss))
                    this_valid_loss, this_kappa = self.validate()
                    self.historical_val_losses.append([iter, this_valid_loss])
                    self.historical_val_kappas.append([iter, this_kappa])
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
                 batch_size, leakiness, center, normalize, amplify, as_grey, num_output_classes):
    runid = "%s-%s-nu%f-cent%i-norm%i-amp%i-grey%i-out%i" % (str(uuid.uuid4())[:8], network, learning_rate, center, normalize, amplify, int(as_grey), num_output_classes)
    print("[INFO] Starting runid %s" % runid)

    with open('network_specs.json') as data_file:
        network = json.load(data_file)[network]
        netspec = network['layers']
        pad = network['pad']
        input_image_size = network['rec_input_size']

    input_image_channels = 1 if as_grey else 3

    image_shape = (input_image_size, input_image_size, input_image_channels)
    model_spec = [{ "type": "INPUT", "size": input_image_size, "channels": input_image_channels}] + netspec

    data_stream = DataStream(image_dir=dataset, image_shape=image_shape, center=center, normalize=normalize, amplify=amplify, num_output_classes=num_output_classes)

    column = VGGNet(data_stream, batch_size, learning_rate, momentum, leakiness, model_spec, num_output_classes=num_output_classes, pad=pad)
    try:
        column.train_column(n_epochs)
    except KeyboardInterrupt:
        print "[ERROR] User terminated Training, saving results"
    except UnsupportedPredictedClasses:
        print "[ERROR] UnsupportedPredictedClasses, saving results"
    column.save(runid)
    save_results(runid, [column.historical_train_losses, column.historical_val_losses, column.historical_val_kappas, column.n_train_batches])

if __name__ == '__main__':
    arg_names = ['command', 'network', 'dataset', 'batch_size', 'center', 'normalize', 'learning_rate', 'momentum', 'leakiness', 'n_epochs', 'amplify', 'as_grey', 'num_output_classes']
    arg = dict(zip(arg_names, sys.argv))

    network = arg.get('network') or 'vgg_mini6'
    dataset = arg.get('dataset') or "data/train/centered_crop/" # data/train_digit/128/ alternative
    batch_size = int(arg.get('batch_size') or 2)
    center = int(arg.get('center') or 0)
    normalize = int(arg.get('normalize') or 0)
    learning_rate = float(arg.get('learning_rate') or 0.01)
    momentum = float(arg.get('momentum') or 0.9)
    leakiness = float(arg.get('leakiness') or 0.01)
    n_epochs = int(arg.get('n_epochs') or 800) # useful to change to 1 for a quick test run
    amplify = int(arg.get('amplify') or 1)
    as_grey = bool(arg.get('as_grey') or 0)
    num_output_classes = int(arg.get('num_output_classes') or 5)

    train_drnet(network=network, learning_rate=learning_rate, momentum=momentum, n_epochs=n_epochs, dataset=dataset, batch_size=batch_size, leakiness=leakiness, center=center, normalize=normalize, amplify=amplify, as_grey=as_grey, num_output_classes=num_output_classes)
