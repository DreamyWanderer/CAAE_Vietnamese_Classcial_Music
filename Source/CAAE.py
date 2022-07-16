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

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

import Config
import Pipline

def construct_Discriminator():

    input_Discriminator = layers.Input( (Config.coding_size,), name = "Input_Discriminator")
    x = input_Discriminator

    for i in range(Config.encoder_dis["depth"] - 1):

        x = layers.Dense(Config.encoder_dis["num_hidden_node"], activation = "selu", name = f"Hidden_layer_{i + 1}")(x)
        x = layers.BatchNormalization(name = f"Batchnorm_{i + 1}")(x)
        x = layers.LeakyReLU(0.2, name = f"LeakyLU_layer_{i + 1}")(x)

    output_Discriminator = layers.Dense(1, activation = "sigmoid")(x)

    Discriminator = models.Model(inputs = input_Discriminator, outputs = output_Discriminator)
    Discriminator.compile(loss = "binary_crossentropy", optimizer = optimizers.Adam(learning_rate = 0.0002) )

    utils.plot_model(Discriminator, show_shapes = True, show_dtype = True, to_file = "Discriminator.png")

    return Discriminator

def construct_Condition_discriminator():

    input_Discriminator = layers.Input( shape = [None, Config.num_row], name = "Input_Discriminator")

    depth = Config.decoder_dis["depth"]
    x = input_Discriminator
    for i in range(depth - 1):
        x = layers.LSTM(Config.encoder_param["num_hidden_node"], return_sequences = True, name = f"LSTM_encoder_{i + 1}")(x)
        x = layers.BatchNormalization(name = f"Batchnorm_{i}")(x)
        x = layers.LeakyReLU(0.2, name = f"LeakyLU_layer_{i}")(x)

    end_LSTM = layers.LSTM(Config.encoder_param["num_hidden_node"], name = f"LSTM_encoder_{depth}")(x)
    x = layers.BatchNormalization(name = f"Batchnorm_{depth}")(end_LSTM)
    x = layers.LeakyReLU(0.2, name = f"LeakyLU_layer_{depth}")(x)

    output_Discriminator = layers.Dense(1, activation = "sigmoid")(x)

    Discriminator = models.Model(inputs = input_Discriminator, outputs = output_Discriminator)
    Discriminator.compile(loss = "binary_crossentropy", optimizer = optimizers.Adam(learning_rate = 0.0002) )

    utils.plot_model(Discriminator, show_shapes = True, show_dtype = True, to_file = "Conditional_discriminator.png")

    return Discriminator    

def train_AAE():

    test_log_dir = CVAE.get_run_logdir()
    writer = summary.create_file_writer(test_log_dir)

    #Prepare all related model
    VAE = CVAE.construct_VAE(False)
    Discriminator = construct_Discriminator()
    Condition_discriminator = construct_Condition_discriminator()
    Encoder = VAE.get_layer(name = "Encoder")
    Decoder = VAE.get_layer(name = "Decoder")

    #Build the first part of CAAE
    AAE_input = Encoder.input
    _, _, x = Encoder(AAE_input)
    x = Discriminator(x)
    AAE = models.Model(AAE_input, x)
    Discriminator.trainable = False
    AAE.compile(loss = "binary_crossentropy", optimizer = optimizers.Adam(learning_rate = 0.0002) )

    utils.plot_model(AAE, show_shapes = True, show_dtype = True, to_file = "AAE.png")

    #Build the second part of CAAE
    input_data = Encoder.input
    input_label = layers.Input( (Config.num_type, ), name = "Label_input")
    x = VAE((input_data, input_label))
    #x = layers.Concatenate(name = "Concat_label", axis = 1)([x, input_label])
    condition_output = Condition_discriminator(x)
    CAAE = models.Model([input_data, input_label], condition_output)
    Condition_discriminator.trainable = False
    CAAE.compile(loss = "binary_crossentropy", optimizer = optimizers.Adam(learning_rate = 0.0002) )

    utils.plot_model(CAAE, show_shapes = True, show_dtype = True, to_file = "CAAE.png")

    #Number of batch in each epoch
    dataset = Pipline.pianoroll_reader_dataset("Dataset\Samples\*.npy")
    print(dataset)
    print(dataset[0])
    print(dataset[1])

    for epoch in range(Config.num_epoch):

        print('Epoch %s:' % epoch)
        x = 0

        for X_batch in dataset:

            #Phase 1: train VAE
            loss_VAE = VAE.train_on_batch(X_batch, X_batch[0])

            #Phase 2: train the discriminator
            real_distribution = tf.random.normal(shape = [Config.batch_size, Config.coding_size], mean = 0.0, stddev = 5.0)
            _, _, fake_distribution = Encoder(X_batch[0])
            X_fake_and_real = tf.concat([fake_distribution, real_distribution], axis = 0)
            label = tf.constant([[0.]] * Config.batch_size + [[1.]] * Config.batch_size)
            loss_Discriminator = Discriminator.train_on_batch(X_fake_and_real, label)

            #Phase 3: train the encoder generator
            label = tf.constant([[1.]] * Config.batch_size)
            loss_encoder = AAE.train_on_batch(X_batch[0], label)

            #Phase 4: train the conditional discriminator
            #Not really the CAAE model
            fake_data = VAE(X_batch)
            Data_fake_and_real = tf.concat([fake_data, X_batch[0]], axis = 0)
            label = tf.constant([[0.]] * Config.batch_size + [[1.]] * Config.batch_size)
            loss_Conditional_discriminator = Condition_discriminator.train_on_batch(Data_fake_and_real, label)

            #Phase 5: train the decoder generator
            label = tf.constant([[1.]] * Config.batch_size)
            loss_decoder = CAAE.train_on_batch(X_batch, label)

            #Log to Tensorboard
            with writer.as_default():
                tf.summary.scalar('Loss VAE', loss_VAE, epoch * num_iter + x)
                tf.summary.scalar('Loss Discriminator', loss_Discriminator, epoch * num_iter + x)
                tf.summary.scalar('Loss Encoder generator', loss_encoder, epoch * num_iter + x)
                tf.summary.scalar('Loss Conditional discriminator', loss_Conditional_discriminator, epoch * num_iter + x)
                tf.summary.scalar('Loss Decoder generator', loss_decoder, epoch * num_iter + x)

            x += 1

        #Save the model after each epoch
        VAE.save("Saved_model\VAE")
        Discriminator.save("Saved_model\Discriminator")
        Condition_discriminator.save("Saved_model\Condition_discriminator")
        AAE.save("Saved_model\AAE")
        CAAE.save("Save_model\CAAE")
        Encoder.save("Saved_model\Encoder")
        Decoder.save("Saved_model\Decoder")

train_AAE()