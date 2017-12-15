from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras
from keras import backend
from keras.datasets import mnist
from keras.models import load_model, model_from_json
from keras.utils import np_utils
from BBalpha_concrete import get_logit_mlp_layers, get_logit_cnn_layers, GenerateMCSamples
from BBalpha_concrete import bbalpha_softmax_cross_entropy_with_mc_logits
from keras.models import Model
from keras.layers import Input, Activation, Dropout, Lambda

import math
import numpy as np
import os
from keras.backend import categorical_crossentropy
import six
import time
import warnings

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

def load_mnist():
    # Get MNIST test data
    #X_train, Y_train, X_test, Y_test = data_mnist()
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    
    return X_train, Y_train, X_test, Y_test

def load_model(path, alpha = 0.5, K_mc = 10, n_epoch = 500, nb_layers = 3, \
               nb_units = 1000, p = 0.5, wd = 1e-6, nb_classes = 10, model_arch = 'mlp', \
               dropout = 'MC', n_mc = 1):

    # Define TF model graph by loading model
    # NOTE: set dropout = True if wanted to test MC dropout
    # else it will use keras dropout and then use p*W for prediction
    #path = '/homes/mlghomes/yl494/proj/dropout/adversarial/'
    
    if model_arch == 'mlp':
        nb_in = 784; input_shape = (nb_in,)
        inp = Input(shape=input_shape)
        layers = get_logit_mlp_layers(nb_layers, nb_units, p, wd, nb_classes, dropout = dropout)
    else:
        img_rows, img_cols = 28, 28; input_shape = (1, img_rows, img_cols)
        inp = Input(shape=input_shape)
        layers = get_logit_cnn_layers(nb_units, p, wd, nb_classes, dropout = dropout)
    # NOTE: should set n_mc = 1 here if dropout is not MC
    if dropout not in ['MC', 'concrete']:
        n_mc = 1
    mc_logits = GenerateMCSamples(inp, layers, n_mc)
    mc_softmax = Activation('softmax')(mc_logits)  # softmax is over last dim
    model = Model(input=inp, output=mc_softmax)
    if dropout != 'concrete': 
        folder = path + model_arch + '_nb_layers_' + str(nb_layers) \
             + '_nb_units_' + str(nb_units) + '_p_' + str(p) + '/'
    else:
        folder = path + model_arch + '_nb_layers_' + str(nb_layers) \
             + '_nb_units_' + str(nb_units) + '_concrete/'

    file_name = folder + 'K_mc_' + str(K_mc) + '_alpha_' + str(alpha)
    model.load_weights(file_name+'_weights.h5', by_name=True)
    print("model loaded from "+file_name+' weights.h5')
    print("Defined TensorFlow model graph.")
    
    return model
    
# evaluation for classification tasks
# Yarin's implementation (parallel MC dropout)
def MC_dropout(model, x, n_mc):
    pred = model(x) # N x K x D
    if n_mc > 1:
        pred = tf.reduce_mean(pred, 1)
    return pred

def batch_eval(sess, tf_inputs, tf_outputs, numpy_inputs, stepsize_ph, stepsize_val, x_original_ph = None, x_original_val = None):
    """
    A helper function that computes a tensor on numpy inputs by batches.
    """
    n = len(numpy_inputs)
    assert n > 0
    assert n == len(tf_inputs)
    m = numpy_inputs[0].shape[0]
    for i in six.moves.xrange(1, n):
        assert numpy_inputs[i].shape[0] == m
    out = []
    for _ in tf_outputs:
        out.append([])
    batch_size = FLAGS.batch_size
    with sess.as_default():
        for start in six.moves.xrange(0, m, batch_size):
            batch = start // batch_size
            if batch % 100 == 0 and batch > 0:
                print("Batch " + str(batch))

            # Compute batch start and end indices
            start = batch * batch_size
            end = start + batch_size
            numpy_input_batches = [numpy_input[start:end] for numpy_input in numpy_inputs]
            cur_batch_size = numpy_input_batches[0].shape[0]
            assert cur_batch_size <= batch_size
            for e in numpy_input_batches:
                assert e.shape[0] == cur_batch_size

            feed_dict = dict(zip(tf_inputs, numpy_input_batches))
            feed_dict[stepsize_ph] = stepsize_val
            feed_dict[keras.backend.learning_phase()] = 0
            if x_original_ph is not None:
                feed_dict[x_original_ph] = x_original_val[start:end]
            numpy_output_batches = sess.run(tf_outputs, feed_dict=feed_dict)
            for e in numpy_output_batches:
                assert e.shape[0] == cur_batch_size, e.shape
            for out_elem, numpy_output_batch in zip(out, numpy_output_batches):
                out_elem.append(numpy_output_batch)

    out = [np.concatenate(x, axis=0) for x in out]
    for e in out:
        assert e.shape[0] == m, e.shape
    return out

def model_eval(sess, x, y, model_MC, X_test, Y_test, Y_target = None, MC = False):
    """
    Compute the accuracy of a TF model on some data
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param model: model output predictions
    :param X_test: numpy array with training inputs
    :param Y_test: numpy array with training outputs
    :return: a float with the accuracy value
    """

    if MC:
        model = tf.reduce_mean(model_MC, 1)
    else:
        model = model_MC

    # Define sympbolic for accuracy
    acc_value = tf.reduce_mean(keras.metrics.categorical_accuracy(y, model))
    entropy_value = -model_MC * tf.log(tf.clip_by_value(model_MC, 1e-8, 1.0 - 1e-8))
    entropy_value = tf.reduce_sum(entropy_value, -1)   

    # Init result var
    accuracy = 0.0
    accuracy_target = 0.0
    
    with sess.as_default():
        # Compute number of batches
        batch_size = 100#FLAGS.batch_size
        nb_batches = int(math.ceil(float(len(X_test)) / batch_size))
        assert nb_batches * batch_size >= len(X_test)

        for batch in range(nb_batches):
            if batch % 100 == 0 and batch > 0:
                print("Batch " + str(batch))

            # Must not use the `batch_indices` function here, because it
            # repeats some examples.
            # It's acceptable to repeat during training, but not eval.
            start = batch * batch_size
            end = min(len(X_test), start + batch_size)
            cur_batch_size = end - start

            # The last batch may be smaller than all others, so we need to
            # account for variable batch size here
            #print(X_test[start:end].shape, Y_test[start:end].shape)
            accuracy += cur_batch_size * acc_value.eval(feed_dict={x: X_test[start:end],
                                            y: Y_test[start:end],
                                            keras.backend.learning_phase(): 0})
            if Y_target is not None:
                accuracy_target += cur_batch_size * acc_value.eval(feed_dict={x: X_test[start:end],
                                            y: Y_target[start:end],
                                            keras.backend.learning_phase(): 0})
            entropy_now = entropy_value.eval(feed_dict={x: X_test[start:end],
                                            y: Y_test[start:end],
                                            keras.backend.learning_phase(): 0})
            if batch == 0:
                entropy = entropy_now
            else:
                entropy = np.concatenate((entropy, entropy_now))
            
        assert end >= len(X_test)
        assert entropy.shape[0] == len(X_test)

        # Divide by number of examples to get final value
        accuracy /= len(X_test)
        accuracy_target /= len(X_test)
        entropy_mean = np.mean(entropy)
        entropy_ste = np.sqrt(np.var(entropy) / len(X_test))
    if Y_target is None:
        return accuracy, entropy_mean, entropy_ste
    else:
        return accuracy, entropy_mean, entropy_ste, accuracy_target
    
def plot_images(ax, images, shape, color = False):
     # finally save to file
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # flip 0 to 1
    images = 1.0 - images
    
    images = reshape_and_tile_images(images, shape, n_cols=len(images))
    if color:
        from matplotlib import cm
        plt.imshow(images, cmap=cm.Greys_r, interpolation='nearest')
    else:
        plt.imshow(images, cmap='Greys')
    ax.axis('off')   

def reshape_and_tile_images(array, shape=(28, 28), n_cols=None):
    if n_cols is None:
        n_cols = int(array.shape[0] / 10)
    n_rows = int(math.ceil(float(array.shape[0])/n_cols))
    if len(shape) == 2:
        order = 'C'
    else:
        order = 'F'
    
    def cell(i, j):
        ind = i*n_cols+j
        if i*n_cols+j < array.shape[0]:
            return array[ind].reshape(shape, order=order)
        else:
            return np.zeros(shape)

    def row(i):
        return np.concatenate([cell(i, j) for j in range(n_cols)], axis=1)

    return np.concatenate([row(i) for i in range(n_rows)], axis=0)

