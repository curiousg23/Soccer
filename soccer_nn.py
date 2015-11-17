import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

# Vanilla NN model
def build_nn(input_var=None):
    # build input layer of unspecified batch size of data pts as row vectors
    l_in = lasagne.layers.InputLayer(shape=(None,8), input_var=input_var)

    # first hidden layer, trying 100 units
    l_1 = lasagne.layers.DenseLayer(l_in, num_units=100, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())

    # classify into 3 categories: Home win, draw, or away win
    l_out = lasagne.layers.DenseLayer(l_1, num_units=3, nonlinearity=lasagne.nonlinearities.softmax)

    return l_out

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def main(num_epochs=500):
    # training data--currently unnormalized/processed
    X_train = np.load('train_data.npy')
    X_val = np.load('val_data.npy')
    Y_train = np.load('train_targets.npy')
    Y_val = np.load('val_targets.npy')

    input_var = T.matrix('inputs', dtype='float64')
    target_var = T.ivector('targets')

    network = build_nn(input_var)

    # use cross-entropy error for classification
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # use SGD with momentum--may want to try out other methods of training
    # try using regular GD as well since dataset is small
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
    # theano function to perform training step using the update scheme and calculated
    # loss above
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        # for GD call with batchsize = size of training set
        for batch in iterate_minibatches(X_train, Y_train, 300, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, Y_val, 20, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

main(30)
