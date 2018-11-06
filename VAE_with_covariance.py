'''Example of VAE on MNIST dataset using MLP

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.

# Reference

[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Multiply, Conv2DTranspose, Reshape
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


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


# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train[:,:,:, np.newaxis]
x_test = x_test[:,:,:, np.newaxis]

# network parameters
input_shape = (x_train.shape[1], x_train.shape[2], 1)
intermediate_dim = 64
batch_size = 128
latent_dim = 2
epochs = 20

# VAE model = encoder + decoder
# build encoder model
input_imgs = Input(shape=input_shape, name='encoder_input')
layer = Conv2D(16, (3, 3), activation='relu', padding='same')(input_imgs)
layer = MaxPooling2D((2, 2), padding='same')(layer)
layer = Conv2D(8, (3, 3), activation='relu', padding='same')(layer)
layer = MaxPooling2D((2, 2), padding='same')(layer)
layer = Conv2D(8, (3, 3), activation='relu', padding='same')(layer)
layer = MaxPooling2D((2, 2), padding='same')(layer)
layer = Flatten()(layer)
layer_to_out = Dense(intermediate_dim, activation='relu')(layer)

z_mean = Dense(latent_dim, name='z_mean')(layer_to_out)
z_covariance = Dense(latent_dim, name='z_log_var')(layer_to_out)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
# Lambda doesn't take keras layer as input, but the 2 tensors returned by encoder
z_sampled = Lambda(sampling, output_shape=(latent_dim,), name='z_resample2out')([z_mean, z_covariance])

# instantiate encoder model
encoder = Model(input_imgs, [z_mean, z_covariance, z_sampled], name='encoder')
encoder.summary()
# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')

x = Dense(intermediate_dim, name='z_mean')(latent_inputs)

x = Dense(4*4*8)(x)

x = Reshape((4, 4, 8))(x)
x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same', strides=(2,2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same', strides=(2,2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='valid')(x)
x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same', strides=(2,2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
output_imgs = Conv2D(1, (3, 3), activation='relu', padding='same')(x)

# instantiate decoder model
decoder = Model(latent_inputs, output_imgs, name='decoder')
decoder.summary()
#plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
output_imgs = decoder(encoder(input_imgs)[2])
vae = Model(input_imgs, output_imgs, name='vae_mlp')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    def loss_function(y_true, y_pred):
        return K.mean(K.binary_crossentropy(y_true, y_pred), axis=[1,2,3]) # axis list flattens the data to 1 dimension


    reconstruction_loss = loss_function(input_imgs, output_imgs)
    reconstruction_loss *= original_dim

    # Kullback–Leibler divergence
    kl_loss = 1 + z_covariance - K.square(z_mean) - K.exp(z_covariance)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5 #
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()

    if args.weights:
        vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
        vae.save_weights('vae_mlp_mnist.h5')

    plot_results(models,
                 data,
                 batch_size=batch_size,
                 model_name="vae_mlp")