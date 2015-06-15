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
    def __init__(self, data_stream, batch_size, init_learning_rate, momentum, leak_alpha, model_spec, num_output_classes, pad=1, params=None):
        self.column_params = [batch_size, init_learning_rate, momentum, leak_alpha, model_spec, pad]
        layer_input_sizes, layer_parameter_counts = self.precompute_layer_sizes(model_spec, pad)
        print "[DEBUG] all_layers input widths: {}".format(layer_input_sizes)
        print "[INFO] Estimated memory usage is %f MB per input image" % round(sum(layer_parameter_counts) * 4e-6 * 3, 2)
        # data setup
        self.ds = data_stream
        self.n_train_batches = len(self.ds.train_dataset) // batch_size
        self.n_valid_batches = len(self.ds.valid_dataset) // batch_size
        self.batches_per_cache_block = self.ds.cache_size // batch_size
        self.num_output_classes = num_output_classes
        self.learning_rate = init_learning_rate
        self.learning_rate_decayed_epochs = []

        self.train_x, self.train_y = self.ds.train_buffer().next()
        self.train_x = theano.shared(lasagne.utils.floatX(self.train_x))
        self.train_y = T.cast(theano.shared(self.train_y), 'int32')

        valid_x, self.valid_y = self.ds.valid_set()
        valid_x = theano.shared(lasagne.utils.floatX(valid_x))
        self.valid_y = T.cast(theano.shared(self.valid_y), 'int32')

        all_layers = self.build_model(model_spec, leak_alpha, pad)

        learning_rate = T.fscalar()
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
            [cache_block_index, learning_rate], loss_train,
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

    def build_model(self, model_spec, leak_alpha, pad):
        print("Building model from JSON...")
        def get_nonlinearity(layer):
            default_nonlinear = "ReLU"  # for all Conv2DLayer, Conv2DCCLayer, and DenseLayer
            req = layer.get("nonlinearity") or default_nonlinear
            return {
                "LReLU": LeakyRectify(1./leak_alpha),
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
                batch_loss = self.train_batch(i, self.learning_rate)
                self.historical_train_losses.append([self.iter, batch_loss])
                yield batch_loss

    def decay_learning_rate(self, patience, factor, limit=2):
        if (len(self.learning_rate_decayed_epochs) < limit and
            max([0] + self.learning_rate_decayed_epochs) + patience < self.epoch): # also skip first 4 epochs

            val_losses = numpy.array(self.historical_val_losses)
            best_val_loss = min(val_losses[:,1])
            last_val_losses = val_losses[-patience:,1]
            if sum(last_val_losses > best_val_loss) == patience:
                self.learning_rate_decayed_epochs.append(self.epoch)
                self.learning_rate = self.learning_rate / factor

    def validate(self, decay_patience, decay_factor, silent=False):
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
        val_loss = numpy.mean(batch_valid_losses)
        self.decay_learning_rate(decay_patience, decay_factor)
        # housekeeping
        self.historical_val_losses.append([self.iter, val_loss])
        self.historical_val_kappas.append([self.iter, kappa])
        print('     epoch %i, minibatch %i/%i, validation error %f %%' %
                          (self.epoch, self.iter + 1 % self.n_train_batches, self.n_train_batches,
                           val_loss * 100.))
        print('     kappa on validation set is: %f' % kappa)
        if not silent and (self.num_output_classes == 5):
            print_confusion_matrix(M)
        return [val_loss, kappa]

    def train_column(self, max_epochs, decay_patience, decay_factor, validations_per_epoch=1):
        print("Training...")
        start_time = time.clock()
        batch_multiple_to_validate = self.n_train_batches // validations_per_epoch
        # reset training state of column
        self.epoch = 0
        self.iter = 0
        self.historical_train_losses = []
        self.historical_val_losses = []
        self.historical_val_kappas = []
        while self.epoch < max_epochs:
            self.epoch += 1
            for batch_train_loss in self.train_epoch():
                self.iter += 1
                if (self.iter + 1) % batch_multiple_to_validate == 0:
                    mins_per_epoch = self.n_train_batches*(time.clock() - start_time)/(self.iter*60.)
                    print('training @ iter = %i @ %.1fm (ETA %.1fm). Cur training error is %f %%' %
                        (self.iter, ((time.clock() - start_time )/60.), mins_per_epoch*(max_epochs-self.epoch), 100*batch_train_loss))
                    self.validate(decay_patience, decay_factor)
                    print('     averaging %f mins per epoch' % mins_per_epoch)

def save_results(filename, multi_params):
    name = filename or 'CNN_%iParams_t%i' % (len(self.params) / 2, int(time.time()))
    print('Saving Results as "%s"...' % name)
    f = open('./results/'+name+'.pkl', 'wb')
    for params in multi_params:
        cPickle.dump(params, f, -1)
    f.close()

def train_drnet(network, init_learning_rate, momentum, max_epochs, dataset,
                 batch_size, leak_alpha, center, normalize, amplify,
                 as_grey, num_output_classes, decay_patience, decay_factor):
    runid = "%s-%s-nu%f-a%i-cent%i-norm%i-amp%i-grey%i-out%i-dp%i-df%i" % (str(uuid.uuid4())[:8], network, init_learning_rate, leak_alpha, center, normalize, amplify, int(as_grey), num_output_classes, decay_patience, decay_factor)
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

    column = VGGNet(data_stream, batch_size, init_learning_rate, momentum, leak_alpha, model_spec, num_output_classes=num_output_classes, pad=pad)
    try:
        column.train_column(max_epochs, decay_patience, decay_factor)
    except KeyboardInterrupt:
        print "[ERROR] User terminated Training, saving results"
    except UnsupportedPredictedClasses:
        print "[ERROR] UnsupportedPredictedClasses, saving results"
    column.save(runid)
    save_results(runid, [[column.historical_train_losses, column.historical_val_losses, column.historical_val_kappas, column.n_train_batches], [column.learning_rate_decayed_epochs]])

if __name__ == '__main__':
    arg_names = ['command', 'network', 'dataset', 'batch_size', 'center', 'normalize', 'init_learning_rate', 'momentum', 'leak_alpha', 'max_epochs', 'amplify', 'as_grey', 'num_output_classes', 'decay_patience', 'decay_factor']
    arg = dict(zip(arg_names, sys.argv))

    network = arg.get('network') or 'vgg_mini7b'
    dataset = arg.get('dataset') or "data/train/centered_crop/" # data/train_digit/128/ alternative
    batch_size = int(arg.get('batch_size') or 128)
    center = int(arg.get('center') or 0)
    normalize = int(arg.get('normalize') or 0)
    init_learning_rate = float(arg.get('init_learning_rate') or 0.01)
    momentum = float(arg.get('momentum') or 0.9)
    leak_alpha = int(arg.get('leak_alpha') or 100)
    max_epochs = int(arg.get('max_epochs') or 100) # useful to change to 1 for a quick test run
    amplify = int(arg.get('amplify') or 1)
    as_grey = bool(arg.get('as_grey') or 0)
    num_output_classes = int(arg.get('num_output_classes') or 5)
    decay_patience = int(arg.get('decay_patience') or 5) # set to max_epochs to avoid decay
    decay_factor = int(arg.get('decay_factor') or 10)

    train_drnet(network=network, init_learning_rate=init_learning_rate, momentum=momentum, max_epochs=max_epochs, dataset=dataset, batch_size=batch_size, leak_alpha=leak_alpha, center=center, normalize=normalize, amplify=amplify, as_grey=as_grey, num_output_classes=num_output_classes, decay_patience=decay_patience, decay_factor=decay_factor)
