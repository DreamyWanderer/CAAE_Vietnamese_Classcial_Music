from ast import Lambda
from pyexpat import model
from turtle import back
from yaml import load
import CVAE

import tensorboard

import tensorflow as tf
from tensorflow import summary
from tensorflow import keras
from keras import datasets
from keras import models
from keras import layers
from keras import utils
from keras import callbacks
from keras import optimizers
from keras import backend

from sklearn.manifold import TSNE
from functools import partial

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

import Config
import Pipeline

class ConcatPianorollAndLabel(layers.Layer):

    def call(self, inputs):

        pianoroll, label = inputs
        pianoroll_shape = tf.shape(pianoroll)
        
        reshape_label = tf.reshape( tf.tile(label, [1, pianoroll_shape[1]]), [pianoroll_shape[0], -1, Config.num_type] )
        return backend.concatenate([pianoroll, reshape_label])

class RandomWeightedAverage(layers.Layer):

    def call(self, inputs):

        input_shape = tf.shape(inputs[0])
        alpha = backend.random_uniform((input_shape[0], 1)) # Or (input_shape[0], 1, 1) shape, does not matter because of broadcasting property of tensor

        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

def grad(y, x):

    V = layers.Lambda(lambda z: backend.gradients(z[0], z[1]), output_shape = [1])([y, x])

    return V

def wasserstein(y_true, y_pred):

    return backend.mean(y_true * y_pred)

#This function receive three argument since the interpolated_samples are not detected until training. y_true argument is only a dummy tensor when compile the Critic WP
def gradient_penalty_loss(y_true, y_pred, interpolated_samples):

    tf.print(y_pred)
    gradients = grad(y_pred, interpolated_samples)[0]
    gradient_square = backend.square(gradients)
    gradient_square_sum = backend.sum(gradient_square, axis = np.arange(1, len(gradient_square.shape)))
    gradient_l2_norm = backend.sqrt(gradient_square_sum)
    gradient_penalty = backend.square(1 - gradient_l2_norm)

    return backend.mean(gradient_penalty)

def construct_Encoder_Critic():

    input_Critic = layers.Input( (Config.coding_size,), name = "Input_critic")
    x = input_Critic

    for i in range(Config.encoder_dis["depth"]):

        x = layers.Dense(Config.encoder_dis["num_hidden_node"], activation = "tanh", name = f"Dense_{i}")(x)

    output_Critic = layers.Dense(1, name = "Output_critic")(x)

    Encoder_critic = models.Model(inputs = input_Critic, outputs = output_Critic, name = "Encoder_critic")
    Encoder_critic.compile(loss = wasserstein, optimizer = optimizers.Adam(learning_rate = 0.0002) )

    utils.plot_model(Encoder_critic, show_shapes = True, show_dtype = True, to_file = "Document\\Model_diagram\\Encoder_critic.png")

    return Encoder_critic

def construct_Decoder_Critic():

    input_Critic = layers.Input(shape = [None, Config.num_feature], name = "Input_critic")
    input_Label = layers.Input(shape = [Config.num_type], name = "Critic_label")

    x = ConcatPianorollAndLabel(name = "Concat_pianoroll_and_label")([input_Critic, input_Label])
    depth = Config.decoder_dis["depth"]
    for i in range(depth - 1):
        x = layers.LSTM(Config.decoder_dis["num_hidden_node"], return_sequences = True, name = f"LSTM_encoder_{i + 1}")(x)

    end_LSTM = layers.LSTM(Config.decoder_dis["num_hidden_node"], name = f"LSTM_encoder_{depth}")(x)
    output_Critic = layers.Dense(1, name = "Output_critic")(end_LSTM)

    Decoder_critic = models.Model(inputs = [input_Critic, input_Label], outputs = output_Critic, name = "Decoder_critic")
    Decoder_critic.compile(loss = wasserstein, optimizer = optimizers.Adam(learning_rate = 0.0002) )

    utils.plot_model(Decoder_critic, show_shapes = True, show_dtype = True, to_file = "Document\\Model_diagram\\Decoder_critic.png")

    return Decoder_critic    

def construct_Encoder_critic_WP(Encoder_critic: models.Model):

    real_input = layers.Input(shape = [Config.coding_size], name = "Real_input")
    fake_input = layers.Input(shape = [Config.coding_size], name = "Fake_input")
    interpolated_img = RandomWeightedAverage(trainable = False)([real_input, fake_input])
    validity_interpolated = Encoder_critic([interpolated_img])
    partial_gp_loss = partial(gradient_penalty_loss, interpolated_samples = interpolated_img)
    partial_gp_loss.__name__ = 'gradient_penalty'

    Encoder_critic_WP = models.Model(inputs = [real_input, fake_input], outputs = validity_interpolated)
    Encoder_critic_WP.compile(loss = partial_gp_loss, optimizer = optimizers.Adam(learning_rate = 0.0002) )

    utils.plot_model(Encoder_critic_WP, show_shapes = True, show_dtype = True, to_file = "Document\\Model_diagram\\Encoder_critic_WP.png")

    return Encoder_critic_WP

def construct_Decoder_critic_WP(Decoder_critic: models.Model):

    real_input = layers.Input(shape = [None, Config.num_feature], name = "Real_input")
    fake_input = layers.Input(shape = [None, Config.num_feature], name = "Fake_input")
    pianoroll_label = layers.Input(shape = [Config.num_type], name = "Label_input")
    interpolated_img = RandomWeightedAverage(trainable = False)([real_input, fake_input])
    validity_interpolated = Decoder_critic([interpolated_img, pianoroll_label])
    partial_gp_loss = partial(gradient_penalty_loss, interpolated_samples = interpolated_img)
    partial_gp_loss.__name__ = 'gradient_penalty'

    Decoder_critic_WP = models.Model(inputs = [real_input, fake_input, pianoroll_label], outputs = validity_interpolated)
    Decoder_critic_WP.compile(loss = partial_gp_loss, optimizer = optimizers.Adam(learning_rate = 0.0002) )

    utils.plot_model(Decoder_critic_WP, show_shapes = True, show_dtype = True, to_file = "Document\\Model_diagram\\Decoder_critic_WP.png")

    return Decoder_critic_WP

def train_loop():

    #Create writer summary for Tensorboard
    test_log_dir = CVAE.get_run_logdir()
    writer_summary = []
    for x in Config.list_summary_writer:
        writer_summary.append( summary.create_file_writer( os.path.join(test_log_dir, x) ) )

    #Prepare all related model
    VAE = CVAE.construct_VAE(False)
    Encoder_critic = construct_Encoder_Critic()
    Decoder_critic = construct_Decoder_Critic()
    Encoder_critic_WP = construct_Encoder_critic_WP(Encoder_critic)
    Decoder_critic_WP = construct_Decoder_critic_WP(Decoder_critic)
    Encoder = VAE.get_layer(name = "Encoder")
    Decoder = VAE.get_layer(name = "Decoder")

    #Build the first part of CAAE (AAE model)
    AAE_input = Encoder.input
    _, _, x = Encoder(AAE_input)
    x = Encoder_critic(x)
    AAE = models.Model(AAE_input, x, name = "AAE")
    Encoder_critic.trainable = False
    AAE.compile(loss = wasserstein, optimizer = optimizers.Adam(learning_rate = 0.0002) )

    utils.plot_model(AAE, show_shapes = True, show_dtype = True, to_file = "Document\\Model_diagram\\AAE.png")

    #Build the second part of CAAE
    input_data = Encoder.input
    input_label = layers.Input( (Config.num_type, ), name = "Label_input")
    x = VAE((input_data, input_label))
    x = Decoder_critic([x, input_label])
    CAAE = models.Model([input_data, input_label], x, name = "VAAE")
    Decoder_critic.trainable = False
    CAAE.compile(loss = wasserstein, optimizer = optimizers.Adam(learning_rate = 0.0002) )

    utils.plot_model(CAAE, show_shapes = True, show_dtype = True, to_file = "Document\\Model_diagram\\CAAE.png")

    # Prepare data and counter
    dataset = Pipeline.dataset
    x = 0

    for epoch in range(Config.num_epoch):

        print('Epoch %s:' % epoch)

        for X_batch in dataset:

            #batch_size = tf.shape(X_batch[0])[0]
            batch_size = 32

            #Phase 1: train VAE
            #loss_VAE = VAE.train_on_batch(X_batch, X_batch[0])

            #Phase 2: train the Encoder critic
            for _ in range(5):

                real_label = tf.constant([[1.]] * batch_size)
                fake_label = tf.constant([[-1.]] * batch_size)
                real_distribution = tf.random.normal(shape = [batch_size, Config.coding_size], mean = 0.0, stddev = 5.0)
                loss_Encoder_critic_real = Encoder_critic.train_on_batch(real_distribution, real_label)
                _, _, fake_distribution = Encoder(X_batch[0])
                loss_Encoder_critic_fake = Encoder_critic.train_on_batch(fake_distribution, fake_label)
                loss_Encoder_gradient_penalty = Encoder_critic_WP.train_on_batch([real_distribution, fake_distribution], real_label) #Real_label target is just for fun since we will use interpolated and 1s label target later
                loss_Encoder_critic = backend.mean(loss_Encoder_critic_real + loss_Encoder_critic_fake + loss_Encoder_gradient_penalty)

            #Phase 3: train the Encoder generator
            loss_encoder = AAE.train_on_batch(X_batch[0], real_label)

            #Phase 4: train the Decoder critic
            for _ in range(5):

                loss_Decoder_critic_real = Decoder_critic.train_on_batch(X_batch, real_label)
                fake_data = VAE(X_batch)
                loss_Decoder_critic_fake = Decoder_critic.train_on_batch([fake_data, X_batch[1]], fake_label)
                loss_Decoder_gradient_penalty = Decoder_critic_WP.train_on_batch([ X_batch[0], fake_data, X_batch[1] ])
                loss_Decoder_critic = backend.mean(loss_Decoder_critic_fake + loss_Decoder_critic_real + loss_Decoder_gradient_penalty)

            #Phase 5: train the Decoder generator
            label = tf.constant([[1.]] * Config.batch_size)
            loss_decoder = CAAE.train_on_batch(X_batch, real_label)

            #Log to Tensorboard
            for x in range(0, 5):
                with writer_summary[x].as_default():
                    tf.summary.scalar('Train Loss', loss_VAE[0], x)
                    tf.summary.scalar('Train Loss', loss_Encoder_critic, x)
                    tf.summary.scalar('Train Loss', loss_encoder, x)
                    tf.summary.scalar('Train Loss', loss_Decoder_critic, x)
                    tf.summary.scalar('Train Loss', loss_decoder, x)
            with writer_summary[5].as_default():
                    tf.summary.scalar('Train accuracy', loss_VAE[1], x)

            x += 1

        #Save the model after each epoch
        VAE.save("Saved_model\VAE")
        Encoder_critic.save("Saved_model\Encoder_critic")
        Decoder_critic.save("Saved_model\Decoder_critic")
        AAE.save("Saved_model\AAE")
        CAAE.save("Save_model\CAAE")
        #Maybe these two model do not need to be saved since they are contained in the VAE model, just saving for sure
        Encoder.save("Saved_model\Encoder")
        Decoder.save("Saved_model\Decoder")

train_loop()