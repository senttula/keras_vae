from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Dropout, Layer, Add, LeakyReLU, Concatenate
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy, mae
from keras.callbacks import Callback
from keras.utils import plot_model, to_categorical
from keras import backend as K
from sklearn.metrics import confusion_matrix

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

x_train = np.concatenate((x_train, x_test))
y_train = np.concatenate((y_train, y_test))


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
classes = 10


def make_models():
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation=LeakyReLU())(inputs)

    z_classes = Dense(classes, name='z_classes', activation='sigmoid')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # make sample with means and variance
    z_sample = Sampling(name='z_sample')([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z_sample, z_classes], name='encoder')

    encoder.summary()

    # build decoder model
    latent_inputs = Input(shape=(latent_dim, ), name='z_sample_')

    x = Dense(intermediate_dim)(latent_inputs)

    y_train_input = Input(shape=(classes, ), name='classes_input')

    x = Concatenate()([x, y_train_input])

    x = Dense(intermediate_dim, activation=LeakyReLU())(x)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    decoder = Model([latent_inputs, y_train_input], outputs, name='decoder')
    decoder.summary()

    # instantiate VAE model
    encoded = encoder(inputs) # take the z_sample+classes as input
    sample = encoded[2]
    outputs = decoder([sample, y_train_input])

    reconstruction_loss = K.abs(inputs-outputs)
    reconstruction_loss *= original_dim

    vae = Model([inputs, y_train_input], outputs, name='vae')

    # variance maximizing
    kl_loss = z_log_var # what we want to maximize
    kl_loss -= K.exp(z_log_var)# penalises from too large z_log_var ==> keeps z_log_var from going infinite
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -.5 # negative to maximize

    class_loss = binary_crossentropy(encoded[3], y_train_input)
    #class_loss *= original_dim

    regualrization = K.sum((K.square(z_mean) * 0.001))

    reconstruction_loss = K.mean(reconstruction_loss)
    kl_loss = K.mean(kl_loss)
    regualrization = K.mean(regualrization)
    class_loss = K.mean(class_loss)

    vae.add_loss(reconstruction_loss)
    vae.add_loss(kl_loss)
    vae.add_loss(regualrization)
    vae.add_loss(class_loss)


    vae.compile(optimizer='adam')
    vae.summary()
    return vae, decoder, encoder


vae, decoder, encoder = make_models()


class save_checkpoint(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch%10==0:
            vae.save_weights('vae_weigths.h5')


checkpointer = save_checkpoint()
callbacks = [checkpointer]

# train the autoencoder

training_input = {'encoder_input': x_train, 'classes_input': y_train}

#vae.load_weights('vae_weigths.h5')

vae.fit(training_input,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=callbacks,
        )

class_prediction = encoder.predict(x_train)[3]
cm = confusion_matrix(np.argmax(y_train, axis=1), np.argmax(class_prediction, axis=1))
cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)

print('confusion matrix for classification')
print(cm)

filename = os.path.join("digits_over_latent.png")
n = 10
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
grid_x = np.linspace(-2, 2, n)
grid_y = np.linspace(-2, 2, n)[::-1]

categories = np.eye(10)

for i, yi in enumerate(grid_y):
    z_sample = np.array([[np.sin(i)*2, yi]])
    for j, xi in enumerate(grid_x):
        z_class = categories[j % 10]
        sample = {'z_sample_': z_sample, 'classes_input': z_class[np.newaxis, :]}
        x_decoded = decoder.predict(sample)
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
