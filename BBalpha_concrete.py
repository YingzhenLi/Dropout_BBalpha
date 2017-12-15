from keras import backend as K
from keras.callbacks import Callback
from keras.datasets import mnist
from keras.layers import Input, Dense, Lambda, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.models import Model
from keras.regularizers import l2
from keras.utils import np_utils
import numpy as np
import os, pickle, sys, time
from concrete_dropout import ConcreteDropout

###################################################################
# aux functions

def Dropout_mc(p):
    layer = Lambda(lambda x: K.dropout(x, p), output_shape=lambda shape: shape)
    return layer

def Identity(p):
    layer = Lambda(lambda x: x, output_shape=lambda shape: shape)
    return layer

def pW(p):
    layer = Lambda(lambda x: x*(1.0-p), output_shape=lambda shape: shape)
    return layer

def apply_layers(inp, layers):
    output = inp
    for layer in layers:
        output = layer(output)
    return output

def GenerateMCSamples(inp, layers, K_mc=20):
    if K_mc == 1:
        return apply_layers(inp, layers)
    output_list = []
    for _ in xrange(K_mc):
        output_list += [apply_layers(inp, layers)]  # THIS IS BAD!!! we create new dense layers at every call!!!!
    def pack_out(output_list):
        #output = K.pack(output_list) # K_mc x nb_batch x nb_classes
        output = K.stack(output_list) # K_mc x nb_batch x nb_classes
        return K.permute_dimensions(output, (1, 0, 2)) # nb_batch x K_mc x nb_classes
    def pack_shape(s):
        s = s[0]
        assert len(s) == 2
        return (s[0], K_mc, s[1])
    out = Lambda(pack_out, output_shape=pack_shape)(output_list)
    return out

# evaluation for classification tasks
def test_MC_dropout(model, X, Y):
    pred = model.predict(X)  # N x K x D
    pred = np.mean(pred, 1)
    acc = np.mean(np.argmax(pred, axis=-1) == np.argmax(Y, axis=-1))
    ll = np.sum(np.log(np.sum(pred * Y, -1)))
    return acc, ll

def logsumexp(x, axis=None):
    x_max = K.max(x, axis=axis, keepdims=True)
    return K.log(K.sum(K.exp(x - x_max), axis=axis, keepdims=True)) + x_max

def bbalpha_softmax_cross_entropy_with_mc_logits(alpha, K_mc):
    alpha = K.cast_to_floatx(alpha)
    def loss(y_true, mc_logits):
        # log(p_ij), p_ij = softmax(logit_ij)
        #assert mc_logits.ndim == 3
        mc_log_softmax = mc_logits - K.max(mc_logits, axis=2, keepdims=True)
        mc_log_softmax = mc_log_softmax - K.log(K.sum(K.exp(mc_log_softmax), axis=2, keepdims=True))
        mc_ll = K.sum(y_true * mc_log_softmax, -1)  # N x K
        #K_mc = mc_ll.get_shape().as_list()[1]	# only for tensorflow
        return - 1. / alpha * (logsumexp(alpha * mc_ll, 1) + K.log(1.0 / K_mc))
    return loss


###################################################################
# the model

def get_logit_mlp_layers(nb_layers, nb_units, p, wd, nb_classes, layers = [], \
                         dropout = 'none'):
    if dropout == 'MC':
        D = Dropout_mc
    if dropout == 'pW':
        D = pW
    if dropout == 'none':
        D = Identity
    if dropout == 'concrete':
        pass
    n_drop = 0 
    for _ in xrange(nb_layers):
        if dropout != 'concrete':
            layers.append(D(p))
            layers.append(Dense(nb_units, activation='relu', W_regularizer=l2(wd)))
        else:
            print 'add concrete dropout, %d' % (n_drop+1)
            n_drop += 1
            N_data = 60000.0	# for MNIST, change later
            wd = 1e-6#l**2. / N_data
            dd = 2. / N_data
            #layers.append(ConcreteDropout(Dense(nb_units, activation='relu', W_regularizer=l2(wd))))
            layers.append(ConcreteDropout(Dense(nb_units, activation='relu'), weight_regularizer=wd, dropout_regularizer=dd))
    if dropout != 'concrete':
        layers.append(D(p))
        layers.append(Dense(nb_classes, W_regularizer=l2(wd)))
    else:
        print 'add concrete dropout, %d' % (n_drop+1)
        n_drop += 1
        N_data = 60000.0	# for MNIST, change later
        wd = 1e-6#l**2. / N_data
        dd = 2. / N_data
        #layers.append(ConcreteDropout(Dense(nb_classes, W_regularizer=l2(wd))))
        layers.append(ConcreteDropout(Dense(nb_classes, activation='relu'), weight_regularizer=wd, dropout_regularizer=dd))
    return layers

def get_logit_cnn_layers(nb_units, p, wd, nb_classes, layers = [], dropout = False):
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)
    
    if dropout == 'MC':
        D = Dropout_mc
    if dropout == 'pW':
        D = pW
    if dropout == 'none':
        D = Identity

    layers.append(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                border_mode='valid', W_regularizer=l2(wd)))
    layers.append(Activation('relu'))
    layers.append(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                W_regularizer=l2(wd)))
    layers.append(Activation('relu'))
    layers.append(MaxPooling2D(pool_size=pool_size))

    layers.append(Flatten())
    layers.append(D(p))
    layers.append(Dense(nb_units, W_regularizer=l2(wd)))
    layers.append(Activation('relu'))
    layers.append(D(p))
    layers.append(Dense(nb_classes, W_regularizer=l2(wd)))
    return layers

