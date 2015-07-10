# -*- coding: utf-8 -*-

# mainstream
import os
import sys
import time
import collections
import uuid
import json
import re
from math import exp
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
# this repo
from my_code.predict_util import QWK, print_confusion_matrix, UnsupportedPredictedClasses
from my_code.data_stream import DataStream
import my_code.train_args as train_args

import pdb

class VGGNet(object):
    def __init__(self, data_stream, batch_size, init_learning_rate, momentum, leak_alpha, model_spec, loss_type, num_output_classes, pad, image_shape):
        self.column_init_args = [batch_size, init_learning_rate, momentum, leak_alpha, model_spec, loss_type, num_output_classes, pad, image_shape]
        layer_input_sizes, layer_parameter_counts = self.precompute_layer_sizes(model_spec, pad)
        print "[DEBUG] all_layers input widths: {}".format(layer_input_sizes)
        print "[INFO] Estimated memory usage is %f MB per input image" % round(sum(layer_parameter_counts) * 4e-6 * 3, 2)
        # data setup
        self.ds = data_stream
        self.batch_size = batch_size
        self.n_test_examples = self.ds.n_test_examples
        self.n_train_batches = self.ds.n_train_batches
        self.n_valid_batches = self.ds.valid_dataset_size // self.batch_size
        self.batches_per_cache_block = self.ds.cache_size // self.batch_size
        self.num_output_classes = num_output_classes
        self.learning_rate = init_learning_rate
        self.learning_rate_decayed_epochs = []
        self.flip_noise = 1
        # both train and test are buffered
        self.x_buffer, self.y_buffer = self.ds.train_buffer().next() # dummy fill in
        self.x_buffer = theano.shared(lasagne.utils.floatX(self.x_buffer))
        self.y_buffer = T.cast(theano.shared(self.y_buffer), 'int32')
        # validation set is not buffered
        valid_x, self.valid_y = self.ds.valid_set()
        valid_x = theano.shared(lasagne.utils.floatX(valid_x))
        self.valid_y = T.cast(theano.shared(self.valid_y), 'int32')
        # network setup
        self.all_layers = self.build_model(model_spec, leak_alpha, pad)

        learning_rate = T.fscalar()
        cache_block_index = T.iscalar('cache_block_index')
        X_batch = T.tensor4('x')
        y_batch = T.ivector('y')
        batch_slice = slice(cache_block_index * self.batch_size, (cache_block_index + 1) * self.batch_size)

        loss_train, loss_valid, pred, raw_out = self.build_loss_predictions(X_batch, y_batch, self.all_layers[-1], loss_type)

        self.params = lasagne.layers.get_all_params(self.all_layers[-1])
        updates = lasagne.updates.nesterov_momentum(loss_train, self.params, learning_rate, momentum)

        print("Compiling...")

        self.train_batch = theano.function(
            [cache_block_index, learning_rate], loss_train,
            updates=updates,
            givens={
                X_batch: self.x_buffer[batch_slice],
                y_batch: self.y_buffer[batch_slice],
            },
        )
        self.validate_batch = theano.function(
            [cache_block_index], [loss_valid, pred],
            givens={
                X_batch: valid_x[batch_slice],
                y_batch: self.valid_y[batch_slice],
            },
        )
        self.test_batch = theano.function(
            [cache_block_index],
            [pred, raw_out],
            givens={
                X_batch: self.x_buffer[batch_slice],
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

    def build_target_label_prediction(self, valid_out, loss_type, K):
        """
        Picks the target vector for each class, as well as a strategy for
        picking the predicted label from the network output
        """
        if re.search('one-hot', loss_type):
            # should be used with softmax out
            identity = numpy.identity(K)
            pred_valid = T.argmax(valid_out, axis=1)
            klass_targets = identity
        elif re.search('nnrank', loss_type):
            # should be used with sigmoid out
            if self.num_output_classes == K:
                nnrank_target = numpy.tril(numpy.ones((K,K)))
                pred_valid = T.sum(T.gt(valid_out, 0.5), axis=1) - 1 # can potentially return -1
            elif self.num_output_classes == (K-1):
                nnrank_target = numpy.array([[0]*(K-1)] + numpy.tril(numpy.ones((K-1,K-1))).tolist())
                # TODO check for discontinuities rather than assuming none with the sum
                # TODO do better than a shared threshold
                pred_valid = T.sum(T.gt(valid_out, 0.5), axis=1)
            klass_targets = nnrank_target
        return(theano.shared(lasagne.utils.floatX((klass_targets))),pred_valid)

    def build_loss_predictions(self, X, y, output, loss_type, K=5):
        train_out = lasagne.layers.get_output(output, X)
        valid_out = lasagne.layers.get_output(output, X, deterministic=True)
        klass_targets, pred_valid = self.build_target_label_prediction(valid_out, loss_type, K)
        target = klass_targets[y]

        if loss_type == 'one-hot': # requires that y has no more than self.num_output_classes classes
            objective = lasagne.objectives.Objective(output, loss_function=lasagne.objectives.categorical_crossentropy)
            loss_train = objective.get_loss(X, target=y)
            loss_valid = objective.get_loss(X, target=y, deterministic=True)
        elif (loss_type == 'one-hot-ce') and (self.num_output_classes == K):
            # manual version of one-hot, totally equivalent (in theory and practice)
            loss_train = -T.mean(T.sum(target*T.log(train_out), axis=1))
            loss_valid = -T.mean(T.sum(target*T.log(valid_out), axis=1))
        elif (loss_type == 'one-hot-re') and (self.num_output_classes == K):
            loss_train = -T.mean(T.sum(target*T.log(train_out) + (1-target)*T.log(1-train_out), axis=1))
            loss_valid = -T.mean(T.sum(target*T.log(valid_out) + (1-target)*T.log(1-valid_out), axis=1))
        elif loss_type == 'nnrank-mse':
            loss_train = T.mean(T.sum((target - train_out)**2, axis=1))
            loss_valid = T.mean(T.sum((target - valid_out)**2, axis=1))
        elif loss_type == 'nnrank-re':
            loss_train = -T.mean(T.sum(target*T.log(train_out) + (1-target)*T.log(1-train_out), axis=1))
            loss_valid = -T.mean(T.sum(target*T.log(valid_out) + (1-target)*T.log(1-valid_out), axis=1))
        else:
            raise ValueError("unsupported loss_type %s for output shape %i" % (loss_type, self.num_output_classes))
        return loss_train, loss_valid, pred_valid, valid_out

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

    def test(self, override_buffer=None, override_num_examples=None):
        """
        Is responsible for moving the data stream into the column's variables
        and aggregating test batches
        :return: predictions (raw probabilities and decision too)
        """
        test_buffer = override_buffer or self.ds.test_buffer
        n_test_examples = override_num_examples or self.n_test_examples
        print("Testing...")
        all_test_predictions = -numpy.ones(n_test_examples, dtype=int)
        all_test_output = -numpy.ones((n_test_examples, self.num_output_classes))
        for x_cache_block, example_idxs in test_buffer():
            self.x_buffer.set_value(lasagne.utils.floatX(x_cache_block), borrow=True)

            for i in xrange(self.batches_per_cache_block):
                batch_slice = slice(i * self.batch_size, (i + 1) * self.batch_size)
                example_idxs_batch = example_idxs[batch_slice]

                pred, raw_out = self.test_batch(i)
                all_test_predictions[example_idxs_batch] = pred
                all_test_output[example_idxs_batch] = raw_out
        return(all_test_predictions,all_test_output)

    def train_epoch(self, decay_patience, decay_factor, decay_limit, noise_decay_start, noise_decay_duration, noise_decay_severity):
        """
        Is responsible for moving the data stream into the column's variables
        Since it mutates the network's weights, the state of the column's
        epoch/iter are updated here as well.

        :return: minibatch training error
        """
        self.epoch += 1
        self.decay_flip_noise(noise_decay_start, noise_decay_duration, noise_decay_severity)
        self.decay_learning_rate(decay_patience, decay_factor, decay_limit)
        for x_cache_block, y_cache_block in self.ds.train_buffer(self.flip_noise):
            self.x_buffer.set_value(lasagne.utils.floatX(x_cache_block), borrow=True)
            self.y_buffer.set_value(y_cache_block, borrow=True)

            for i in xrange(self.batches_per_cache_block):
                self.iter += 1
                batch_loss = self.train_batch(i, self.learning_rate)
                self.historical_train_losses.append([self.iter, batch_loss])
                yield batch_loss

    def decay_learning_rate(self, patience, factor, limit):
        if (len(self.learning_rate_decayed_epochs) < limit and
            max([0] + self.learning_rate_decayed_epochs) + patience < self.epoch): # also skip first n epochs (n = patience)

            val_losses = numpy.array(self.historical_val_losses)
            best_val_loss = min(val_losses[:,1])
            last_val_losses = val_losses[-patience:,1]
            if sum(last_val_losses > best_val_loss) == patience:
                self.learning_rate_decayed_epochs.append(self.epoch)
                self.learning_rate = self.learning_rate / factor

    def decay_flip_noise(self, noise_decay_start, noise_decay_duration, noise_decay_severity):
        if (noise_decay_start > 0): # decay noise at a specific time
            schedule = [exp(1)**(-(x*noise_decay_severity)/float(noise_decay_duration)) for x in range(noise_decay_duration)]
            sched_idx = max(0, self.epoch - noise_decay_start)
            if (sched_idx < len(schedule)):
                self.flip_noise = schedule[sched_idx]
            else:
                self.flip_noise = 0
        else:
            # TODO could start decaying flip noise when val loss plateaus
            raise NotImplementedError()

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
        val_loss = numpy.mean(batch_valid_losses)
        # housekeeping
        self.historical_val_losses.append([self.iter, val_loss])
        self.historical_val_kappas.append([self.iter, kappa])
        print('     epoch %i, minibatch %i/%i, validation error %f %%' %
                          (self.epoch, (self.iter + 1) % self.n_train_batches, self.n_train_batches,
                           val_loss * 100.))
        print('     kappa on validation set is: %f' % kappa)
        if not silent:
            print_confusion_matrix(M)
        return [val_loss, kappa]

    def train(self, max_epochs, decay_patience, decay_factor, decay_limit, noise_decay_start, noise_decay_duration, noise_decay_severity, validations_per_epoch):
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
            for batch_train_loss in self.train_epoch(decay_patience, decay_factor, decay_limit, noise_decay_start, noise_decay_duration, noise_decay_severity):
                if (self.iter + 1) % batch_multiple_to_validate == 0:
                    mins_per_epoch = self.n_train_batches*(time.clock() - start_time)/(self.iter*60.)
                    print('training @ iter = %i @ %.1fm (ETA %.1fm). Cur training error is %f %%' %
                        (self.iter, ((time.clock() - start_time )/60.), mins_per_epoch*(max_epochs-self.epoch), 100*batch_train_loss))
                    self.validate()
                    print('     averaging %f mins per epoch' % mins_per_epoch)

    def save(self, filename=None):
        name = filename or 'CNN_%iLayers_t%i' % (len(self.params) / 2, int(time.time()))
        print('Saving Model as "%s"...' % name)
        f = open('./models/'+name+'.pkl', 'wb')
        cPickle.dump(self.column_init_args, f, -1)
        # only need to save last layer params to restore them later
        cPickle.dump(lasagne.layers.get_all_param_values(self.all_layers[-1]), f, -1)
        f.close()

    def restore(self, filepath):
        print("Restoring...")
        f = open(filepath)
        all_saves = []
        while True:
            try:
                all_saves.append(cPickle.load(f))
            except EOFError:
                break
        lasagne.layers.set_all_param_values(self.all_layers[-1], all_saves[-1])
        f.close()

def save_results(filename, multi_params):
    name = filename or 'CNN_%iParams_t%i' % (len(self.params) / 2, int(time.time()))
    print('Saving Results as "%s"...' % name)
    f = open('./results/'+name+'.pkl', 'wb')
    for params in multi_params:
        cPickle.dump(params, f, -1)
    f.close()

def load_model_specs(network, as_grey):
    with open('network_specs.json') as data_file:
        network = json.load(data_file)[network]
        netspec = network['layers']
        pad = network['pad']
        input_image_size = network['rec_input_size']
    input_image_channels = 1 if as_grey else 3
    image_shape = (input_image_size, input_image_size, input_image_channels)
    model_spec = [{ "type": "INPUT", "size": input_image_size, "channels": input_image_channels}] + netspec
    return(model_spec, image_shape, pad)

def init_and_train(network, init_learning_rate, momentum, max_epochs, train_dataset,
                 batch_size, leak_alpha, center, normalize, amplify,
                 as_grey, num_output_classes, decay_patience, decay_factor,
                 decay_limit, loss_type, validations_per_epoch, train_flip,
                 shuffle, test_dataset, random_seed, valid_dataset_size,
                 noise_decay_start, noise_decay_duration, noise_decay_severity, valid_flip, test_flip):
    runid = "%s-%s-%s-nu%f-a%i-cent%i-norm%i-amp%i-grey%i-out%i-dp%i-df%i" % (str(uuid.uuid4())[:8], network, loss_type, init_learning_rate, leak_alpha, center, normalize, amplify, int(as_grey), num_output_classes, decay_patience, decay_factor)
    print("[INFO] Starting runid %s" % runid)

    model_spec, image_shape, pad = load_model_specs(network, as_grey)

    data_stream = DataStream(train_image_dir=train_dataset, batch_size=batch_size, image_shape=image_shape, center=center, normalize=normalize, amplify=amplify, train_flip=train_flip, shuffle=shuffle, test_image_dir=test_dataset, random_seed=random_seed, valid_dataset_size=valid_dataset_size, valid_flip=valid_flip, test_flip=test_flip)
    column = VGGNet(data_stream, batch_size, init_learning_rate, momentum, leak_alpha, model_spec, loss_type, num_output_classes, pad, image_shape)
    try:
        column.train(max_epochs, decay_patience, decay_factor, decay_limit, noise_decay_start, noise_decay_duration, noise_decay_severity, validations_per_epoch)
    except KeyboardInterrupt:
        print "[ERROR] User terminated Training, saving results"
    except UnsupportedPredictedClasses as e:
        print "[ERROR] UnsupportedPredictedClasses {}, saving results".format(e.args[0])
    column.save(runid)
    save_results(runid, [[column.historical_train_losses, column.historical_val_losses, column.historical_val_kappas, column.n_train_batches], [column.learning_rate_decayed_epochs]])
    print(time.strftime("Finished at %H:%M:%S on %Y-%m-%d"))

if __name__ == '__main__':
    _ = train_args.get()

    init_and_train(network=_.network,
                init_learning_rate=_.learning_rate,
                momentum=_.momentum,
                max_epochs=_.max_epochs,
                train_dataset=_.train_dataset,
                batch_size=_.batch_size,
                leak_alpha=_.alpha,
                center=_.center,
                normalize=_.normalize,
                amplify=_.amplify,
                as_grey=_.as_grey,
                num_output_classes=_.output_classes,
                decay_patience=_.decay_patience,
                decay_factor=_.decay_factor,
                decay_limit=_.decay_limit,
                loss_type=_.loss_type,
                validations_per_epoch=_.validations_per_epoch,
                train_flip=_.train_flip,
                shuffle=_.shuffle,
                test_dataset=_.test_dataset,
                random_seed=_.random_seed,
                valid_dataset_size=_.valid_dataset_size,
                noise_decay_start=_.noise_decay_start,
                noise_decay_duration=_.noise_decay_duration,
                noise_decay_severity=_.noise_decay_severity,
                valid_flip=_.valid_flip,
                test_flip=_.test_flip)
