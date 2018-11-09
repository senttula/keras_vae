'''
modified from https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Dropout, Layer, Add, LeakyReLU
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.callbacks import Callback
from keras.utils import plot_model
from keras import backend as K

from tensorflow import random_shuffle

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()

class Sampling(Add):
    # merging Layer
    # inputs: mean, variance
    # output: random sample with these parameters
    def _merge_function(self, inputs):
        z_mean = inputs[0]
        z_log_var = inputs[1]

        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]

        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

# network parameters
input_shape = (original_dim, )

intermediate_dim = 32
batch_size = 128
latent_dim = 2
epochs = 50

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation=LeakyReLU())(inputs)
x = Dense(intermediate_dim, activation=LeakyReLU())(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
# make sample with means and variance
z_sample = Sampling(name='z_sample')([z_mean, z_log_var])

encoder = Model(inputs, [z_mean, z_log_var, z_sample], name='encoder')
encoder.summary()

# build decoder model
latent_inputs = Input(shape=(latent_dim, ), name='z_sample')
x = Dense(intermediate_dim, activation=LeakyReLU())(latent_inputs)
x = Dense(intermediate_dim, activation=LeakyReLU())(x)
outputs = Dense(original_dim, activation='sigmoid')(x)

decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# instantiate VAE model
sample = encoder(inputs)[2]  # take the z_sample as input
outputs = decoder(sample)
vae = Model(inputs, outputs, name='vae')


reconstruction_loss = mse(inputs, outputs)
reconstruction_loss *= original_dim

# variance maximizing
kl_loss = z_log_var # what we want to maximize
kl_loss -= K.exp(z_log_var)# penalises from too large z_log_var ==> keeps z_log_var from going infinite
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -.5 # negative to maximize

# separating point means
difference = z_mean-K.reverse(z_mean, axes=[1]) # pair latent points randomly, reverse is random enough
total_distance = (K.sum(K.abs(difference), axis=1))+1  # L1 distance the pairs and sum
mean_diversity = -K.log(total_distance)  # negative to maximize distance, log keeps means sensible

regualrization = K.sum((K.square(z_mean)+K.abs(z_mean)))*0.001#  keeps means from growing indefinitely ==> regualrization

vae_loss = K.mean(reconstruction_loss + kl_loss + regualrization)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()

# train the autoencoder

vae.fit(x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None),
        verbose=1)

vae.save_weights('vae_mlp_mnist.h5')

plot_results((encoder, decoder),
             (x_test, y_test),
             batch_size=batch_size,
             model_name="vae_mlp")
