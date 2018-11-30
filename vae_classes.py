'''
modified from https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Dropout, Layer, Add, LeakyReLU, Concatenate
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.callbacks import Callback
from keras.utils import plot_model, to_categorical
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

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
epochs = 5
batch_separation = False
classes = 10


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

def md():
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation=LeakyReLU())(inputs)
    x = Dense(intermediate_dim, activation=LeakyReLU())(x)

    z_classes = Dense(classes, name='z_classes', activation='sigmoid')(x)

    #x = Dense((intermediate_dim+latent_dim)//2)(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # make sample with means and variance
    z_sample = Sampling(name='z_sample')([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z_sample, z_classes], name='encoder')
    encoder.summary()

    # build decoder model
    latent_inputs = Input(shape=(latent_dim, ), name='z_sample_')
    #x = Dense((intermediate_dim+latent_dim)//2)(latent_inputs)
    x = Dense(intermediate_dim)(latent_inputs)

    #classes_inputs = Input(shape=(classes, ), name='z_classes')
    y_train_input = Input(shape=(classes, ), name='classes_input')

    x = Concatenate()([latent_inputs, y_train_input])

    x = Dense(intermediate_dim, activation=LeakyReLU())(x)
    x = Dense(intermediate_dim, activation=LeakyReLU())(x)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    decoder = Model([latent_inputs, y_train_input], outputs, name='decoder')
    decoder.summary()

    # instantiate VAE model
    encoded = encoder(inputs)
    sample = encoded[2:4]  # take the z_sample+classes as input
    outputs = decoder(sample)



    reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss *= original_dim






    vae = Model([inputs, y_train_input], outputs, name='vae')



    # variance maximizing
    kl_loss = z_log_var # what we want to maximize
    kl_loss -= K.exp(z_log_var)# penalises from too large z_log_var ==> keeps z_log_var from going infinite
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -.5 # negative to maximize

    class_loss = binary_crossentropy(encoded[3], y_train_input)

    regualrization = K.sum((K.square(z_mean) * 0.001))

    #vae_loss = K.mean(reconstruction_loss + kl_loss + regualrization + class_loss)

    reconstruction_loss = K.mean(reconstruction_loss)
    kl_loss = K.mean(kl_loss)
    regualrization = K.mean(regualrization)
    class_loss = K.mean(class_loss)

    vae.add_loss(reconstruction_loss)
    vae.add_loss(kl_loss)
    vae.add_loss(regualrization)
    vae.add_loss(class_loss)

    print(reconstruction_loss)
    print(kl_loss)
    print(regualrization)
    print(class_loss)

    vae.compile(optimizer='adam')
    vae.summary()
    return vae, decoder

vae, decoder = md()

#vae.load_weights('vae_weigths.h5')

class save_checkpoint(Callback):
    def set_model(self, model):
        pass

    def set_all(self, model, encoder_predicts):
        self.model = model
        self.encoder_predicts = encoder_predicts
        self.epoch = 0
        self.saved_images = []
        self.save_frequency = 100

    def on_epoch_end(self, epoch, logs=None):
        titlenumber = str(0.0)
        filename = os.path.join(temp_folder, titlenumber + '_means.png')
        self.save_image(filename, titlenumber)
        self.saved_images.append(filename)

    def save_image(self, filename, titlenumber):
        return
        z_mean, _, _, _ = encoder.predict([x_test, y_test], batch_size=batch_size)
        plt.figure(figsize=(12, 10))
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.title('epoch: '+titlenumber)
        plt.axis([-8, 8, -8, 8])
        plt.savefig(filename)

callbacks = []
# train the autoencoder

print(x_train.shape)
print(y_train.shape)

ip = {'encoder_input': x_train, 'classes_input': y_train}
print(vae.evaluate(ip,
        batch_size=batch_size,
        #validation_data=(x_test, [x_test, y_test]),
        verbose=2,))
quit()

vae.fit(ip,
        epochs=10,
        batch_size=batch_size,
        #validation_data=(x_test, [x_test, y_test]),
        verbose=2,
        callbacks=callbacks)

vae.save_weights('vae_weigths.h5')

filename = os.path.join("digits_over_latent.png")
# display a 30x30 2D manifold of digits
n = 10
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates corresponding to the 2D plot
# of digit classes in the latent space
grid_x = np.linspace(-2, 2, n)
grid_y = np.linspace(-2, 2, n)[::-1]

categories = np.eye(10)

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[(((xi-yi)*2)%2-1)*2, xi-yi-2]])
        z_class = categories[j%10]
        ip = {'z_sample_': z_sample, 'classes_input': z_class[np.newaxis, :]}
        x_decoded = decoder.predict(ip)
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

