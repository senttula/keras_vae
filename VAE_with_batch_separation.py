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

def make_gif():
    try:
        import imageio
        import datetime
        images = []
        for filename in checkpointer.saved_images:
            images.append(imageio.imread(filename))
        output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%d-%H-%M-%S')
        imageio.mimwrite(output_file, images, duration=0.1)
    except e:
        print(e)

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
batch_separation = True


temp_folder = 'temp_images'
os.makedirs(temp_folder, exist_ok=True)
for the_file in os.listdir(temp_folder):
    file_path = os.path.join(temp_folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)

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





if batch_separation:
    # separating point means
    difference = z_mean - K.reverse(z_mean, axes=[1])  # pair latent points randomly, reverse is random enough
    # total_distance = (K.sum(K.abs(difference), axis=1))+1  # L1 distance the pairs and sum
    total_distance = K.sqrt(K.sum(K.square(difference), axis=1))  # distance the pairs and sum
    mean_diversity = -K.log(total_distance)  # negative to maximize distance, log keeps means sensible
    regualrization = K.sum((K.abs(z_mean) * 0.005 + K.square(z_mean) * 0.001))#  keeps means from growing indefinitely ==> regualrization
    vae_loss = K.mean(reconstruction_loss + kl_loss + regualrization + mean_diversity)
else:
    regualrization = K.sum((K.abs(z_mean) * 0.001 + K.square(z_mean) * 0.001))
    vae_loss = K.mean(reconstruction_loss + kl_loss + regualrization)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()

vae.load_weights('vae_weigths.h5')


class save_checkpoint(Callback):
    def set_model(self, model):
        pass

    def set_all(self, model, encoder_predicts):
        self.model = model
        self.encoder_predicts = encoder_predicts
        self.epoch = 0
        self.saved_images = []
        self.save_frequency = 100

    def on_train_begin(self, logs=None):
        z_mean, _, _ = self.model.predict(x_test)
        self.encoder_predicts = np.concatenate((self.encoder_predicts, z_mean[np.newaxis, :, :]), axis=0)
        titlenumber = str(0.0)
        filename = os.path.join(temp_folder,  titlenumber + '_means.png')
        self.save_image(filename, titlenumber)
        self.saved_images.append(filename)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch+1
        if epoch==5:
            self.save_frequency = 230
        if epoch==10:
            self.save_frequency = 1000

    def on_batch_end(self, batch, logs=None):
        if batch%self.save_frequency ==0:
            z_mean, _, _ = self.model.predict(x_test)
            self.encoder_predicts = np.concatenate((self.encoder_predicts, z_mean[np.newaxis, :, :]), axis=0)
            titlenumber = str(self.epoch+round(batch/468, 1))
            filename = os.path.join(temp_folder,  titlenumber + '_means.png')
            self.save_image(filename, titlenumber)
            self.saved_images.append(filename)

    def save_image(self, filename, titlenumber):
        z_mean, _, _ = encoder.predict(x_test,
                                       batch_size=batch_size)
        plt.figure(figsize=(12, 10))
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.title('epoch: '+titlenumber)
        plt.axis([-8, 8, -8, 8])
        plt.savefig(filename)

encoder.compile(optimizer='adam', loss='mse')
z_mean, _, _ = encoder.predict(x_test)
encoder_predicts = z_mean[np.newaxis, :, :]
checkpointer = save_checkpoint()
checkpointer.set_all(encoder, encoder_predicts)

callbacks = [checkpointer]
# train the autoencoder
vae.fit(x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None),
        verbose=2,
        callbacks=callbacks)

print(checkpointer.encoder_predicts.shape)

#vae.save_weights('vae_weigths.h5')


make_gif()
