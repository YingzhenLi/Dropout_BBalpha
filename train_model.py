'''
Copyright 2017, Yingzhen Li and Yarin Gal, All rights reserved.
Please consider citing the ICML 2017 paper if using any of this code for your research:

Yingzhen Li and Yarin Gal.
Dropout inference in Bayesian neural networks with alpha-divergences.
International Conference on Machine Learning (ICML), 2017.

'''

from keras import backend as K
from keras.callbacks import Callback
from keras.datasets import mnist
from keras.layers import Input, Dense, Lambda, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.models import Model
from keras.regularizers import l2
from keras.utils import np_utils
import numpy as np
import os, pickle, sys, time
from BBalpha_dropout import *

nb_epoch = 500
nb_batch = 128
nb_classes = 10
nb_test_mc = 100
wd = 1e-6

if len(sys.argv) == 7:
    K_mc = int(sys.argv[1])
    alpha = float(sys.argv[2])
    nb_layers = int(sys.argv[3])
    nb_units = int(sys.argv[4])
    p = float(sys.argv[5])
    model_arch = sys.argv[6]

folder = 'save/' + model_arch + '_nb_layers_' + str(nb_layers) + '_nb_units_' + str(nb_units) + '_p_' + str(p) + '/'
if not os.path.exists('save/'):
    os.makedirs('save/')
if not os.path.exists(folder):
    os.makedirs(folder)

file_name = folder + 'K_mc_' + str(K_mc) + '_alpha_' + str(alpha) + '.obj'
print file_name


if model_arch == 'mlp':
    nb_in = 784
    input_shape = (nb_in,)
else:
    img_rows, img_cols = 28, 28
    input_shape = (1, img_rows, img_cols)


###################################################################
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, *input_shape)
X_test = X_test.reshape(10000, *input_shape)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

###################################################################
# compile model

inp = Input(shape=input_shape)
if model_arch == 'mlp':
    layers = get_logit_mlp_layers(nb_layers, nb_units, p, wd, nb_classes, dropout = 'MC')
else:
    layers = get_logit_cnn_layers(nb_units, p, wd, nb_classes, dropout = 'MC')
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
    with open(file_name, 'wb') as f:
        pickle.dump(evals, f)

# save model for future test
file_name = folder + 'K_mc_' + str(K_mc) + '_alpha_' + str(alpha)
model.save_weights(file_name+'_weights.h5')
print 'model weights saved to file ' + file_name + '_weights.h5'


