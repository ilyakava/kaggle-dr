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
# from lasagne import layers, lasagne.nonlinearities
# from lasagne.nonlinearities import LeakyRectify
# this repo
from my_code.predict_util import QWK, print_confusion_matrix, UnsupportedPredictedClasses
from my_code.data_stream import DataStream
import my_code.train_args as train_args
from my_code.test.block_designer_test import ACTUAL_TRAIN_DR_PROPORTIONS
from my_code.layers import Fold4xChannelsLayer, Fold4xBatchesLayer, Unfold4xBatchesLayer

import pdb

class VGGNet(object):
    def __init__(self, data_stream, batch_size, init_learning_rate, momentum,
                 leak_alpha, model_spec, loss_type, num_output_classes, pad,
                 image_shape, filter_shape, cuda_convnet=1, runid=None):
        global lasagne
        self.column_init_args = [batch_size, init_learning_rate, momentum, leak_alpha, model_spec, loss_type, num_output_classes, pad, image_shape]
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
        self.runid = runid

        self.cuda_convnet = cuda_convnet
        if self.cuda_convnet:
            import lasagne.layers.cuda_convnet # will crash if theano device is not the GPU
        self.convOp = lasagne.layers.cuda_convnet.Conv2DCCLayer if self.cuda_convnet else lasagne.layers.Conv2DLayer
        self.maxOp = lasagne.layers.cuda_convnet.MaxPool2DCCLayer if self.cuda_convnet else lasagne.layers.MaxPool2DLayer
        # both train and test are buffered
        self.x_buffer, self.y_buffer = self.ds.train_buffer().next() # dummy fill in
        self.x_buffer = theano.shared(lasagne.utils.floatX(self.x_buffer))
        self.y_buffer = T.cast(theano.shared(self.y_buffer), 'int32')
        # validation set is not buffered
        valid_x, self.valid_y = self.ds.valid_set()
        valid_x = theano.shared(lasagne.utils.floatX(valid_x))
        self.valid_y = T.cast(theano.shared(self.valid_y), 'int32')
        # network setup
        self.all_layers = self.build_model(model_spec, leak_alpha, pad, filter_shape)
        widths, weights, memory = self.widths_and_total_num_parameters(filter_shape)
        print "[DEBUG] all_layers input widths: {}".format(widths)
        print "[DEBUG] all_layers weights: {}".format(weights)
        print "[INFO]  %f Million parameters in model" % round(sum(weights) * 1e-6, 2)
        fudge_factor = 2.73 # empirically determined on M2090
        print "[INFO] Estimated memory usage is %f MB per input image" % round(memory * 4e-6 * fudge_factor, 2)

        learning_rate = T.fscalar()
        cache_block_index = T.iscalar('cache_block_index')
        X_batch = T.tensor4('x')
        y_batch = T.ivector('y')
        batch_slice = slice(cache_block_index * self.batch_size, (cache_block_index + 1) * self.batch_size)

        self.params = lasagne.layers.get_all_params(self.all_layers[-1])
        loss_train, loss_valid, pred, raw_out = self.build_loss_predictions(X_batch, y_batch, self.all_layers[-1], loss_type, K=self.ds.K)
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

        layer_idx_of_interest = 5
        l2_activations = T.mean(lasagne.layers.get_output(self.all_layers[layer_idx_of_interest], X_batch) ** 2)
        dream_updates = lasagne.updates.sgd(-l2_activations, [X_batch], learning_rate)
        self.dream_batch = theano.function(
            [cache_block_index],
            [dream_updates.values()[0]],
            updates=dream_updates,
            givens={
                X_batch: self.x_buffer[batch_slice],
            }
        )



    def widths_and_total_num_parameters(self, filter_shape):
        # http://cs231n.github.io/convolutional-networks/
        _c01 = slice(0,-1) if filter_shape == 'c01b' else slice(1,5)
        _01 = slice(1,-1) if filter_shape == 'c01b' else slice(2,5)
        _c = 0 if filter_shape == 'c01b' else 1

        memory = numpy.prod(self.all_layers[0].output_shape[1:]) # pre 1st dimshuffle, i.e. bc01 no matter what
        weights = []
        widths = [self.all_layers[0].output_shape[-1]] # pre 1st dimshuffle, i.e. bc01 no matter what
        for i in range(1,len(self.all_layers)):
            l = self.all_layers[i]
            p = self.all_layers[i-1]
            if type(l) is lasagne.layers.dense.DenseLayer:
                memory += l.output_shape[-1]
                widths.append(l.output_shape[-1])
                weights.append(numpy.prod(p.output_shape[1:])*l.output_shape[-1]) # after 2nd dimshuffle, i.e. bc01 no matter what
            elif type(l) is lasagne.layers.pool.FeaturePoolLayer:
                memory += l.output_shape[-1]
            elif type(l) is self.convOp:
                memory += numpy.prod(l.output_shape[_c01])
                weights.append(numpy.prod(l.get_W_shape()[_01])*p.output_shape[_c]*l.output_shape[_c])
            elif type(l) is self.maxOp:
                memory += numpy.prod(l.output_shape[_c01])
                widths.append(l.output_shape[2])
        return(widths,weights,memory)

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

    def build_loss_predictions(self, X, y, output, loss_type, K):
        train_out = lasagne.layers.get_output(output, X)
        valid_out = lasagne.layers.get_output(output, X, deterministic=True)
        klass_targets, pred_valid = self.build_target_label_prediction(valid_out, loss_type, K)
        target = klass_targets[y]
        if re.search('re', loss_type): # relative entropy
            loss_valid = -T.mean(T.sum(target*T.log(valid_out) + (1-target)*T.log(1-valid_out), axis=1))

        if loss_type == 'one-hot': # requires that y has no more than self.num_output_classes classes
            objective = lasagne.objectives.Objective(output, loss_function=lasagne.objectives.categorical_crossentropy)
            loss_train = objective.get_loss(X, target=y)
            loss_valid = objective.get_loss(X, target=y, deterministic=True)
        elif (loss_type == 'one-hot-ce') and (self.num_output_classes == K): # cross entropy
            # manual version of one-hot, totally equivalent (in theory and practice)
            loss_train = -T.mean(T.sum(target*T.log(train_out), axis=1))
            loss_valid = -T.mean(T.sum(target*T.log(valid_out), axis=1))
        elif (loss_type == 'one-hot-re') and (self.num_output_classes == K): # relative entropy
            loss_train = -T.mean(T.sum(target*T.log(train_out) + (1-target)*T.log(1-train_out), axis=1))
        elif loss_type == 'nnrank-mse': # mean squared error
            loss_train = T.mean(T.sum((target - train_out)**2, axis=1))
            loss_valid = T.mean(T.sum((target - valid_out)**2, axis=1))
        elif loss_type == 'nnrank-re':
            loss_train = -T.mean(T.sum(target*T.log(train_out) + (1-target)*T.log(1-train_out), axis=1))
        elif loss_type == 'nnrank-re-l1':
            lambda_ = 0.01
            l1 = (lambda_ / (2*self.batch_size)) * sum([abs(params).sum() for params in self.params])
            loss_train = -T.mean(T.sum(target*T.log(train_out) + (1-target)*T.log(1-train_out), axis=1)) + l1
        elif loss_type == 'nnrank-re-l2':
            lambda_ = 0.01
            l2 = (lambda_ / (2*self.batch_size)) * sum([(params ** 2).sum() for params in self.params])
            loss_train = -T.mean(T.sum(target*T.log(train_out) + (1-target)*T.log(1-train_out), axis=1)) + l2
        elif loss_type == 'nnrank-re-popw': # population weighted
            # member of each class is scaled down/up as until batch is uniformly made
            # up of gradients of all classes
            avg_klass_size = self.batch_size / float(K)
            klass_proportions = avg_klass_size / T.sum(T.eq(y.dimshuffle(0,'x'), numpy.repeat(numpy.array([list(range(K))]), self.batch_size,axis=0)),0)
            klass_weights = klass_proportions[y]
            loss_train = -T.mean(klass_weights * T.sum(target*T.log(train_out) + (1-target)*T.log(1-train_out), axis=1))
        elif loss_type == 'nnrank-re-pathw': # pathologically weighted
            # non-pathological classes are weighted down so that they contribute
            # just as much as pathological classes to gradient
            non_pathological_cases = T.eq(y, numpy.zeros(self.batch_size))
            num_non_pathological_cases = T.sum(non_pathological_cases) + numpy.spacing(1)
            non_pathological_cases_scale = ((self.batch_size - num_non_pathological_cases)/num_non_pathological_cases) * non_pathological_cases
            klass_weights = non_pathological_cases_scale + T.neq(y, numpy.zeros(self.batch_size)) # fill in with ones for pathologicals
            loss_train = -T.mean(klass_weights * T.sum(target*T.log(train_out) + (1-target)*T.log(1-train_out), axis=1))
        elif (loss_type == 'nnrank-re-sqrtkappa-sym') and (self.num_output_classes == K-1):
            dx = numpy.ones((K,1)) * numpy.arange(K)
            dy = dx.transpose()
            d = numpy.sqrt(abs(dx - dy))
            overestimate_penalty = numpy.triu(d[:,1:]) / (numpy.spacing(1) + (numpy.sum(numpy.triu(d[:,1:]), axis=1)/(numpy.arange(K)[::-1]+numpy.spacing(1))).reshape((5,1)))
            underestimate_penalty = overestimate_penalty[::-1, ::-1]
            overestimate_penalty = theano.shared(lasagne.utils.floatX((overestimate_penalty)))
            underestimate_penalty = theano.shared(lasagne.utils.floatX((underestimate_penalty)))
            loss_train = -T.mean(T.sum((underestimate_penalty[y])*T.log(train_out) + (overestimate_penalty[y])*T.log(1-train_out), axis=1))
        elif (loss_type == 'nnrank-re-kappa-sym') and (self.num_output_classes == K-1):
            dx = numpy.ones((K,1)) * numpy.arange(K)
            dy = dx.transpose()
            d = abs(dx - dy)
            overestimate_penalty = numpy.triu(d[:,1:]) / (numpy.spacing(1) + (numpy.sum(numpy.triu(d[:,1:]), axis=1)/(numpy.arange(K)[::-1]+numpy.spacing(1))).reshape((5,1)))
            underestimate_penalty = overestimate_penalty[::-1, ::-1]
            overestimate_penalty = theano.shared(lasagne.utils.floatX((overestimate_penalty)))
            underestimate_penalty = theano.shared(lasagne.utils.floatX((underestimate_penalty)))
            loss_train = -T.mean(T.sum((underestimate_penalty[y])*T.log(train_out) + (overestimate_penalty[y])*T.log(1-train_out), axis=1))
        elif (loss_type == 'nnrank-re-poly2kappa-sym') and (self.num_output_classes == K-1):
            dx = numpy.ones((K,1)) * numpy.arange(K)
            dy = dx.transpose()
            d = abs(dx - dy) + (dx - dy)**2
            overestimate_penalty = numpy.triu(d[:,1:]) / (numpy.spacing(1) + (numpy.sum(numpy.triu(d[:,1:]), axis=1)/(numpy.arange(K)[::-1]+numpy.spacing(1))).reshape((5,1)))
            underestimate_penalty = overestimate_penalty[::-1, ::-1]
            overestimate_penalty = theano.shared(lasagne.utils.floatX((overestimate_penalty)))
            underestimate_penalty = theano.shared(lasagne.utils.floatX((underestimate_penalty)))
            loss_train = -T.mean(T.sum((underestimate_penalty[y])*T.log(train_out) + (overestimate_penalty[y])*T.log(1-train_out), axis=1))
        elif (loss_type == 'nnrank-re-custkappa-sym') and (self.num_output_classes == K-1):
            # target(3,4) + poly2 hybrid
            overestimate_penalty = numpy.array([[ 1  ,  1  ,  1  ,  1  ],
                                                [ 0. ,  1  ,  1  ,  1  ],
                                                [ 0. ,  0. ,  0.5,  1.5],
                                                [ 0. ,  0. ,  0. ,  1. ],
                                                [ 0. ,  0. ,  0. ,  0. ]])
            underestimate_penalty = overestimate_penalty[::-1, ::-1]
            overestimate_penalty = theano.shared(lasagne.utils.floatX((overestimate_penalty)))
            underestimate_penalty = theano.shared(lasagne.utils.floatX((underestimate_penalty)))
            loss_train = -T.mean(T.sum((underestimate_penalty[y])*T.log(train_out) + (overestimate_penalty[y])*T.log(1-train_out), axis=1))
        elif (loss_type == 'nnrank-re-cust2kappa-sym') and (self.num_output_classes == K-1):
            # kappa(3,4), poly2 hybrid
            overestimate_penalty = numpy.array([[ 0.4,  0.8,  1.2,  1.6],
                                                [ 0. ,  0.5,  1. ,  1.5],
                                                [ 0. ,  0. ,  0.5,  1.5],
                                                [ 0. ,  0. ,  0. ,  1. ],
                                                [ 0. ,  0. ,  0. ,  0. ]])
            underestimate_penalty = overestimate_penalty[::-1, ::-1]
            overestimate_penalty = theano.shared(lasagne.utils.floatX((overestimate_penalty)))
            underestimate_penalty = theano.shared(lasagne.utils.floatX((underestimate_penalty)))
            loss_train = -T.mean(T.sum((underestimate_penalty[y])*T.log(train_out) + (overestimate_penalty[y])*T.log(1-train_out), axis=1))
        else:
            raise ValueError("unsupported loss_type %s for output shape %i" % (loss_type, self.num_output_classes))
        return loss_train, loss_valid, pred_valid, valid_out

    def build_model(self, model_spec, leak_alpha, pad, filter_shape):
        print("Building model from JSON...")
        def get_nonlinearity(layer):
            default_nonlinear = "ReLU"  # for all Conv2DLayer, Conv2DCCLayer, and DenseLayer
            req = layer.get("nonlinearity") or default_nonlinear
            return {
                "LReLU": lasagne.nonlinearities.LeakyRectify(1./leak_alpha),
                "None": None,
                "sigmoid": lasagne.nonlinearities.sigmoid,
                "ReLU": lasagne.nonlinearities.rectify,
                "softmax": lasagne.nonlinearities.softmax,
                "tanh": lasagne.nonlinearities.tanh
            }[req]
        def get_init(layer):
            default_init = "GlorotUniform" # for both Conv2DLayer and DenseLayer (Conv2DCCLayer is None)
            req = layer.get("init") or default_init
            return {
                "Normal": lasagne.init.Normal(),
                "Orthogonal": lasagne.init.Orthogonal(gain='relu'),
                "GlorotUniform": lasagne.init.GlorotUniform()
            }[req]
        def get_custom(layer):
            return {
                "Fold4xChannelsLayer": Fold4xChannelsLayer,
                "Fold4xBatchesLayer": Fold4xBatchesLayer,
                "Unfold4xBatchesLayer": Unfold4xBatchesLayer
            }[layer]

        all_layers = [lasagne.layers.InputLayer(shape=(self.batch_size, model_spec[0]["channels"], model_spec[0]["size"], model_spec[0]["size"]))]
        if filter_shape == 'c01b':
            all_layers.append(lasagne.layers.cuda_convnet.ShuffleBC01ToC01BLayer(all_layers[-1]))
        dimshuffle = False if filter_shape == 'c01b' else True
        kwargs = {'dimshuffle': dimshuffle} if self.cuda_convnet else {}
        for i in xrange(1,len(model_spec)):
            cs = model_spec[i] # current spec
            if cs["type"] == "CONV":
                border_mode = 'full' if pad else 'valid'
                if cs.get("dropout"):
                    all_layers.append(lasagne.layers.DropoutLayer(all_layers[-1], p=cs["dropout"]))
                all_layers.append(self.convOp(all_layers[-1],
                                    num_filters=cs["num_filters"],
                                    filter_size=(cs["filter_size"], cs["filter_size"]),
                                    border_mode=border_mode,
                                    W=get_init(cs),
                                    nonlinearity=get_nonlinearity(cs),
                                    **kwargs))
                if cs.get("pool_size"):
                    all_layers.append(self.maxOp(all_layers[-1],
                                        pool_size=(cs["pool_size"], cs["pool_size"]),
                                        stride=(cs["pool_stride"], cs["pool_stride"]),
                                        **kwargs))
            elif cs["type"] == "FC":
                if (model_spec[i-1]["type"] == "CONV") and (filter_shape == 'c01b'):
                    all_layers.append(lasagne.layers.cuda_convnet.ShuffleC01BToBC01Layer(all_layers[-1]))
                if cs.get("dropout"):
                    all_layers.append(lasagne.layers.DropoutLayer(all_layers[-1], p=cs["dropout"]))
                all_layers.append(lasagne.layers.DenseLayer(all_layers[-1],
                                   num_units=cs["num_units"],
                                   W=get_init(cs),
                                   nonlinearity=get_nonlinearity(cs)))
                if cs.get("pool_size"):
                    all_layers.append(lasagne.layers.FeaturePoolLayer(all_layers[-1], cs["pool_size"]))
            elif cs["type"] == "OUTPUT":
                if cs.get("dropout"):
                    all_layers.append(lasagne.layers.DropoutLayer(all_layers[-1], p=cs["dropout"]))
                all_layers.append(lasagne.layers.DenseLayer(all_layers[-1],
                                   num_units=self.num_output_classes,
                                   W=get_init(cs),
                                   nonlinearity=get_nonlinearity(cs)))
            elif cs["type"] == "CUSTOM":
                custom_layer = get_custom(cs["name"])
                all_layers.append(custom_layer(all_layers[-1]))
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

    def save_progress(self):
        val_losses = numpy.array(self.historical_val_losses)
        val_kappas = numpy.array(self.historical_val_kappas)
        if min(val_losses[:,1]) == val_losses[-1,1]:
            self.save("%s-bestval" % (self.runid))
        if max(val_kappas[:,1]) == val_kappas[-1,1]:
            self.save("%s-bestkappa" % (self.runid))
        save_results(self.runid, [[self.historical_train_losses, self.historical_val_losses, self.historical_val_kappas, self.n_train_batches], [self.learning_rate_decayed_epochs]])

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
        [kappa, M] = QWK(self.valid_y.get_value(borrow=True), numpy.array(valid_predictions), K=self.ds.K)
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
        batch_multiple_to_validate = max(1, int(self.n_train_batches / float(validations_per_epoch)))
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
                    self.save_progress()
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

def load_model_specs(network, as_grey, override_input_size):
    with open('network_specs.json') as data_file:
        network = json.load(data_file)[network]
        netspec = network['layers']
        pad = network['pad']
        input_image_size = override_input_size or network['rec_input_size']
    input_image_channels = 1 if as_grey else 3
    image_shape = (input_image_size, input_image_size, input_image_channels)
    model_spec = [{ "type": "INPUT", "size": input_image_size, "channels": input_image_channels}] + netspec
    return(model_spec, image_shape, pad)

def init_and_train(network, init_learning_rate, momentum, max_epochs, train_dataset,
                 train_labels_csv_path, batch_size, leak_alpha, center, normalize, amplify,
                 as_grey, num_output_classes, decay_patience, decay_factor,
                 decay_limit, loss_type, validations_per_epoch, train_flip,
                 shuffle, test_dataset, random_seed, valid_dataset_size,
                 noise_decay_start, noise_decay_duration, noise_decay_severity,
                 valid_flip, test_flip, sample_class, custom_distribution,
                 train_color_cast, valid_color_cast, test_color_cast,
                 color_cast_range, override_input_size, model_file, filter_shape,
                 cache_size_factor, cuda_convnet, pre_train_crop, train_crop, valid_test_crop):
    runid = "%s-%s-%s" % (str(uuid.uuid4())[:8], network, loss_type)
    print("[INFO] Starting runid %s" % runid)
    if custom_distribution and sample_class: # lame hardcode
        print("[INFO] %.2f current epochs equals 1 BlockDesigner epoch" % ((274.0*numpy.array(custom_distribution)) / numpy.array(ACTUAL_TRAIN_DR_PROPORTIONS))[sample_class])

    model_spec, image_shape, pad = load_model_specs(network, as_grey, override_input_size)
    data_stream = DataStream(train_image_dir=train_dataset, train_labels_csv_path=train_labels_csv_path, image_shape=image_shape, cache_size_factor=cache_size_factor, batch_size=batch_size, center=center, normalize=normalize, amplify=amplify, train_flip=train_flip, shuffle=shuffle, test_image_dir=test_dataset, random_seed=random_seed, valid_dataset_size=valid_dataset_size, valid_flip=valid_flip, test_flip=test_flip, sample_class=sample_class, custom_distribution=custom_distribution, train_color_cast=train_color_cast, valid_color_cast=valid_color_cast, test_color_cast=test_color_cast, color_cast_range=color_cast_range, pre_train_crop=pre_train_crop, train_crop=train_crop, valid_test_crop=valid_test_crop)

    if model_file:
        f = open(model_file)
        batch_size, init_learning_rate, momentum, leak_alpha, model_spec, loss_type, num_output_classes, pad, image_shape = cPickle.load(f)
        f.close()

        column = VGGNet(data_stream, batch_size, init_learning_rate, momentum, leak_alpha, model_spec, loss_type, num_output_classes, pad, image_shape, filter_shape, cuda_convnet=cuda_convnet, runid=runid)
        column.restore(model_file)
    else:
        column = VGGNet(data_stream, batch_size, init_learning_rate, momentum, leak_alpha, model_spec, loss_type, num_output_classes, pad, image_shape, filter_shape, cuda_convnet=cuda_convnet, runid=runid)

    try:
        column.train(max_epochs, decay_patience, decay_factor, decay_limit, noise_decay_start, noise_decay_duration, noise_decay_severity, validations_per_epoch)
    except KeyboardInterrupt:
        print "[ERROR] User terminated Training, saving results"
    except UnsupportedPredictedClasses as e:
        print "[ERROR] UnsupportedPredictedClasses {}, saving results".format(e.args[0])
    column.save("%s_final" % runid)
    save_results(runid, [[column.historical_train_losses, column.historical_val_losses, column.historical_val_kappas, column.n_train_batches], [column.learning_rate_decayed_epochs]])
    print(time.strftime("Finished at %H:%M:%S on %Y-%m-%d"))

if __name__ == '__main__':
    _ = train_args.get()

    init_and_train(network=_.network,
                init_learning_rate=_.learning_rate,
                momentum=_.momentum,
                max_epochs=_.max_epochs,
                train_dataset=_.train_dataset,
                train_labels_csv_path=_.train_labels_csv_path,
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
                test_flip=_.test_flip,
                sample_class=_.sample_class,
                custom_distribution=_.custom_distribution,
                train_color_cast=_.train_color_cast,
                valid_color_cast=_.valid_color_cast,
                test_color_cast=_.test_color_cast,
                color_cast_range=_.color_cast_range,
                override_input_size=_.override_input_size,
                model_file=_.model_file,
                filter_shape=_.filter_shape,
                cache_size_factor=_.cache_size_factor,
                cuda_convnet=_.cuda_convnet,
                pre_train_crop=_.pre_train_crop,
                train_crop=_.train_crop,
                valid_test_crop=_.valid_test_crop)
