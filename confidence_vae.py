'''
modified from https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input, Dense, Add
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt

# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

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
epochs = 25

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
x = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
# make sample with means and variance
z_sample = Sampling(name='z_sample')([z_mean, z_log_var])

encoder = Model(inputs, [z_mean, z_log_var, z_sample], name='encoder')
encoder.summary()

# build decoder model
latent_inputs = Input(shape=(latent_dim, ), name='z_sample')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
x = Dense(intermediate_dim, activation='relu')(x)
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
kl_loss = z_log_var  # what we want to maximize
kl_loss -= K.exp(z_log_var)  # penalises from too large z_log_var ==> keeps z_log_var from going infinite
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -.5  # negative to maximize

regualrization = K.sum(K.abs(z_mean) * 0.002)
vae_loss = K.mean(reconstruction_loss + kl_loss + regualrization)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()

# train the autoencoder
vae.fit(x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None),
        verbose=2)

vae_loss = K.mean(reconstruction_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

figure = plt.figure(figsize=(12, 10))
losses = []
x_train = x_train.reshape((x_train.shape[0], 1, -1))

for i in x_train:
    # evaluate loss for each sample
    # TODO Loop takes long, is there a keras way to get each loss separately without loop
    sample_loss = vae.evaluate(i, verbose=0)
    losses.append(sample_loss)

losses = np.array(losses)
loss_args = np.argsort(losses)

number_of_samples = 8 * 2

worst_losses_indices = np.argpartition(losses, -number_of_samples)[-number_of_samples:]
best_losses_indices = np.argpartition(losses, number_of_samples)[-number_of_samples:]

worst_samples = x_train[worst_losses_indices]
worst_samples = worst_samples.reshape(-1, image_size)

best_samples = x_train[best_losses_indices]
best_samples = best_samples.reshape(-1, image_size)

middle_index = best_samples.shape[0]//2
best_samples = np.concatenate((best_samples[:middle_index], best_samples[middle_index:]), axis=1)
worst_samples = np.concatenate((worst_samples[:middle_index], worst_samples[middle_index:]), axis=1)

plt.subplot(121)
plt.imshow(best_samples)
plt.title('Best samples')
plt.subplot(122)
plt.imshow(worst_samples)
plt.title('Worst samples')

plt.savefig('sample_differences.png')









