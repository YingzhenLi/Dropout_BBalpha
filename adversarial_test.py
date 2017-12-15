from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from loading_utils import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import pickle

import sys, os
from attacks import fgsm

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'mnist.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')

def main(argv=None):
    """
    MNIST cleverhans tutorial
    :return:
    """

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to 'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = load_mnist()

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 784))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Define TF model graph by loading model
    #path = '/homes/mlghomes/yl494/proj/dropout/adversarial/test/'
    path = 'save/'
    alpha = 0.0; K_mc = 10; n_epoch = 500; nb_layers = 3
    nb_units = 1000; p = 0.5; wd = 1e-6; nb_classes = 10
    model_arch = 'mlp'
    dropout = 'concrete'
    n_mc = 10
    model = load_model(path, alpha, K_mc, n_epoch, nb_layers, \
                       nb_units, p, wd, nb_classes, model_arch, \
                       dropout, n_mc)
    
    # construct prediction tensor
    if dropout in ['MC', 'concrete']:
        predictions = MC_dropout(model, x, n_mc = n_mc)
        string = ' (with MC, %d samples)' % n_mc
    else:
        predictions = model(x)
        string = ' (w/out MC)'

    # first check model accuracy on test data
    accuracy, entropy, _ = model_eval(sess, x, y, predictions, X_test, Y_test)
    print('Test accuracy on test data: ' + str(accuracy) + string)   
    print('Test entropy on test data: ' + str(entropy) + string)   

    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    stepsize = tf.placeholder(tf.float32, shape=())
    adv_x = fgsm(x, predictions, eps=stepsize, clip_min = 0.0, clip_max = 1.0)
    
    accuracy_list = []
    entropy_mean_list = []
    entropy_ste_list = []
    stepsize_list = np.arange(0.0, 0.501, 0.02)
    vis_images = []
    for val in stepsize_list:
        X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test], \
                             stepsize_ph = stepsize, stepsize_val = val)
        # Evaluate the accuracy of the MNIST model on adversarial examples
        accuracy, entropy_mean, entropy_ste = \
            model_eval(sess, x, y, predictions, X_test_adv, Y_test)
        accuracy_list.append(accuracy)
        entropy_mean_list.append(entropy_mean)
        entropy_ste_list.append(entropy_ste)
        vis_images.append(X_test_adv[0])
   
    print('Test accuracy on adversarial data: ' + str(accuracy) + string)   
    print('Test entropy on adversarial data: ' + str(entropy_mean) + string)   
 
    accuracy_list = np.array(accuracy_list)
    entropy_mean_list = np.array(entropy_mean_list)
    f, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(stepsize_list, accuracy_list, 'b-')
    ax[1].plot(stepsize_list, entropy_mean_list, 'r-')
    ax[1].fill_between(stepsize_list, entropy_mean_list - entropy_ste_list, \
        entropy_mean_list + entropy_ste_list, color='r', alpha=0.3)
    plot_images(ax[2], np.array(vis_images), shape = (28, 28))
    plt.savefig('untargeted_attack.png', format='png')
    
    # save result
    filename = model_arch + '_nb_layers_' + str(nb_layers) \
             + '_nb_units_' + str(nb_units) + '_p_' + str(p) + \
             '_K_mc_' + str(K_mc) + '_alpha_' + str(alpha)
    if dropout == 'MC':
        filename = filename + '_n_mc_' + str(n_mc)
    elif dropout == 'pW':
        filename = filename + '_pW'
    elif dropout == 'concrete':
        filename = filename + '_concrete'
    else:
        filename = filename + '_no_drop'
    savepath = 'adv_test_results/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    with open(savepath + filename, 'w') as f:
        pickle.dump([stepsize_list, accuracy_list, entropy_mean_list, \
                     entropy_ste_list, vis_images], f)
    print('evaluation results saved in ' + savepath + filename)

if __name__ == '__main__':
    app.run()
