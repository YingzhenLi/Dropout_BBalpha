# Dropout + BB-alpha for detecting adversarial examples

Thank you for your interest in our paper:
[Yingzhen Li](yingzhenli.net) and 
[Yarin Gal](yarin.co)

[Dropout inference in Bayesian neural networks with alpha-divergences](http://proceedings.mlr.press/v70/li17a/li17a.pdf)

International Conference on Machine Learning (ICML), 2017

Please consider citing the paper when any of the material is used for your research.

Contributions: Yarin wrote most of the functions in [BBalpha_dropout.py](BBalpha_dropout.py), and Yingzhen (me) derived the loss function and implemented the adversarial attack experiments.

## how to use this code for your research

I've got quite a few emails on how to incorporate our method into their Keras code. Thus here I also provide a template file, and you can follow the comments inside to plugin your favourate model and dropout method.

template file: [template_model.py](template_model.py)

## repreduce the adversarial attack example

We also provide the adversarial attack detection codes. The attack implementation was adapted from the [cleverhans](http://www.cleverhans.io/) toolbox (version 1.0), and I rewrited the targeted attack to make it an iterative method.

To reproduce the experiments, first train a model on mnist:

python train_model.py <K_mc> <alpha> <nb_layers> <nb_units> <p> <model_arch>

with K_mc the number of MC samples for training, nb_layers the number of layers of the NN, nb_units the number of hidden units in each hidden layer, p the dropout rate (between 0 and 1), and model_arch = mlp or cnn

This will train a model on MNIST data for 500 iterations and save the model. Then to test the FGSM attack, run

python adversarial_test.py 

and change the settings in that python file to pick a saved model for testing. If wanted to see targeted attack, run instead

python adversarial_test_targeted.py

Both files will produce a png file visualising the accuracy, predictive entropy, and some samples of the adversarial image (aligned with the x-axis in the plots).
