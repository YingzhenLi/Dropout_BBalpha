'''
This is a template file for using BB-alpha dropout method for your own
research. I marked all the part you need to write your own code with
# TODO.

Our current implementation of multi MC sample dropout is
not very efficient, so if you have found any better implementation on
this please contact Yarin Gal or Yingzhen Li (emails on their websites).

This implementation works for Keras 1.2.0, Theano 0.8.2, Tensorflow 0.11.0

Copyright 2017, Yingzhen Li and Yarin Gal, All rights reserved.
Please consider citing the ICML 2017 paper if using any of this code for your research:

Yingzhen Li and Yarin Gal.
Dropout inference in Bayesian neural networks with alpha-divergences.
International Conference on Machine Learning (ICML), 2017.

'''

from keras import backend as K
from keras.callbacks import Callback
from keras.layers import Input, Dense, Lambda, Activation, Flatten, \
                         Convolution2D, MaxPooling2D
from keras.models import Model
from keras.regularizers import l2
from keras.utils import np_utils
import numpy as np
import os, pickle, sys, time
from BBalpha_dropout import *


###################################################################
# load your data
(X_train, y_train), (X_test, y_test) = LOAD_YOUR_DATA() # TODO
# define input shape
input_shape = (dimX, )  # TODO: if your input x is a vector
input_shape = (1, height, width)    # TODO: if x is an image, theano
input_shape = (height, width, 1)    # TODO: if x is an image, tensorflow
# preprocess your data
PREPROCESSING   # TODO
# convert class vectors to binary class matrices, in classification
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

###################################################################
# build your Keras model
input_shape = YOUR_DATA_SHAPE   # TODO
inp = Input(shape=input_shape)
# define your dropout technique, e.g. see BBalpha_dropout.py
# or use other methods like concrete dropout that also has a
# Keras implementation.
def your_dropout(args):     # TODO
    layer = Lambda(lambda x: YOUR_DROPOUT_METHOD(x, args), \
                   output_shape = lambda shape: shape)
# define a list of neural network layers
# here you should use your dropout function between layers
# and make sure your dropout function does sample different dropout
# masks for different calls
layers = DEFINE_YOUR_NN_LAYERS_FROM_BOTTOM_TO_TOP   # TODO

###################################################################
# compile model, here we provide the cross-entropy loss
# for other loss see BBalpha_dropout.py file and rewrite accordingly
mc_logits = GenerateMCSamples(inp, layers, K_mc)
if alpha != 0:
    model = Model(input=inp, output=mc_logits)
    model.compile(optimizer='sgd', loss=bbalpha_softmax_cross_entropy_with_mc_logits(alpha),
                  metrics=['accuracy'])
else:
    mc_softmax = Activation('softmax')(mc_logits)  # softmax is over last dim
    model = Model(input=inp, output=mc_softmax)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

mc_logits = GenerateMCSamples(inp, layers, nb_test_mc)
mc_softmax = Activation('softmax')(mc_logits)  # softmax is over last dim
test_model = Model(input=inp, output=mc_softmax)


###################################################################
# train the model
# since alpha divergence methods require K > 1 MC samples, we need to replicate Y data
Y_train_dup = np.squeeze(np.concatenate(K_mc * [Y_train[:, None]], axis=1)) # N x K x D
Y_test_dup = np.squeeze(np.concatenate(K_mc * [Y_test[:, None]], axis=1)) # N x K x D

evals = {'acc_approx': [], 'acc': [], 'll': [], 'time': [], 'train_acc': [], 'train_loss': [],
         'nb_layers': nb_layers, 'nb_units': nb_units}
for i in xrange(100):
    tic = time.clock()
    train_loss = model.fit(X_train, Y_train_dup, verbose=1,
                           batch_size=nb_batch, nb_epoch=nb_epoch//100)
    toc = time.clock()
    evals['acc_approx'] += [model.evaluate(X_test, Y_test_dup, verbose=0)[1]]
    acc, ll = test_MC_dropout(test_model, X_test, Y_test)
    evals['acc'] += [acc]
    evals['ll'] += [ll]
    evals['time'] += [toc - tic]
    evals['train_acc'] += [train_loss.history['acc'][-1]]
    evals['train_loss'] += [train_loss.history['loss'][-1]]
    with open(file_name, 'wb') as f:    # TODO
        pickle.dump(evals, f)

# save model for future test
SAVE_YOUR_MODEL()   # TODO

