from yaml import load
import CVAE
from CVAE import dataset, dataset_test, num_epoch, batch_size, coding_size, num_type

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

def construct_Discriminator():

    input_Discriminator = layers.Input( (coding_size,), name = "Input_Discriminator")
    
    hidden_layer_1 = layers.Dense(150, activation = "selu", name = "Hidden_layer_1")(input_Discriminator)
    batchnorm_layer_1 = layers.BatchNormalization(name = "Batchnorm_1")(hidden_layer_1)
    leakyLU_layer_1 = layers.LeakyReLU(0.2, name = "LeakyLU_layer_1")(batchnorm_layer_1)
    hidden_layer_2 = layers.Dense(150, activation = "selu", name = "Hidden_layer_2")(leakyLU_layer_1)
    batchnorm_layer_2 = layers.BatchNormalization(name = "Batchnorm_2")(hidden_layer_2)
    leakyLU_layer_2 = layers.LeakyReLU(0.2, name = "Leaky_layer_2")(batchnorm_layer_2)
    output_Discriminator = layers.Dense(1, activation = "sigmoid")(leakyLU_layer_2)

    Discriminator = models.Model(inputs = input_Discriminator, outputs = output_Discriminator)
    Discriminator.compile(loss = "binary_crossentropy", optimizer = optimizers.Adam(learning_rate = 0.0002) )

    utils.plot_model(Discriminator, show_shapes = True, show_dtype = True, to_file = "Discriminator.png")

    return Discriminator

def construct_Condition_discriminator():

    input_Discriminator = layers.Input( shape = [28, 28], name = "Input_Discriminator")
    flatten = layers.Flatten(name = "Flatten_layer")(input_Discriminator)
    hidden_layer_1 = layers.Dense(150, activation = "selu", name = "Hidden_layer_1")(input_Discriminator)
    batchnorm_layer_1 = layers.BatchNormalization(name = "Batchnorm_1")(hidden_layer_1)
    leakyLU_layer_1 = layers.LeakyReLU(0.2, name = "LeakyLU_layer_1")(batchnorm_layer_1)
    hidden_layer_2 = layers.Dense(150, activation = "selu", name = "Hidden_layer_2")(leakyLU_layer_1)
    batchnorm_layer_2 = layers.BatchNormalization(name = "Batchnorm_2")(hidden_layer_2)
    leakyLU_layer_2 = layers.LeakyReLU(0.2, name = "Leaky_layer_2")(batchnorm_layer_2)
    output_Discriminator = layers.Dense(1, activation = "sigmoid")(leakyLU_layer_2)

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
    input_label = layers.Input( (num_type, ), name = "Label_input")
    x = VAE((input_data, input_label))
    #x = layers.Concatenate(name = "Concat_label", axis = 1)([x, input_label])
    condition_output = Condition_discriminator(x)
    CAAE = models.Model([input_data, input_label], condition_output)
    Condition_discriminator.trainable = False
    CAAE.compile(loss = "binary_crossentropy", optimizer = optimizers.Adam(learning_rate = 0.0002) )

    utils.plot_model(CAAE, show_shapes = True, show_dtype = True, to_file = "CAAE.png")

    #Number of batch in each epoch
    num_iter = len(dataset)

    for epoch in range(num_epoch):

        print('Epoch %s:' % epoch)
        x = 0

        for X_batch in dataset:

            #Phase 1: train VAE
            loss_VAE = VAE.train_on_batch(X_batch, X_batch[0])

            #Phase 2: train the discriminator
            real_distribution = tf.random.normal(shape = [batch_size, coding_size], mean = 0.0, stddev = 5.0)
            _, _, fake_distribution = Encoder(X_batch[0])
            X_fake_and_real = tf.concat([fake_distribution, real_distribution], axis = 0)
            label = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            loss_Discriminator = Discriminator.train_on_batch(X_fake_and_real, label)

            #Phase 3: train the encoder generator
            label = tf.constant([[1.]] * batch_size)
            loss_encoder = AAE.train_on_batch(X_batch[0], label)

            #Phase 4: train the conditional discriminator
            #Not really the CAAE model
            fake_data = VAE(X_batch)
            Data_fake_and_real = tf.concat([fake_data, X_batch[0]], axis = 0)
            label = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            loss_Conditional_discriminator = Condition_discriminator.train_on_batch(Data_fake_and_real, label)

            #Phase 5: train the decoder generator
            label = tf.constant([[1.]] * batch_size)
            loss_decoder = CAAE.train_on_batch(X_batch, label)

            #Log to Tensorboard
            with writer.as_default():
                tf.summary.scalar('Loss VAE', loss_VAE, epoch * num_iter + x)
                tf.summary.scalar('Loss Discriminator', loss_Discriminator, epoch * num_iter + x)
                tf.summary.scalar('Loss Encoder generator', loss_encoder, epoch * num_iter + x)
                tf.summary.scalar('Loss Conditional discriminator', loss_Conditional_discriminator, epoch * num_iter + x)
                tf.summary.scalar('Loss Decoder generator', loss_decoder, epoch * num_iter + x)
                tf.summary.scalar('Total loss', loss_VAE + loss_Discriminator + loss_encoder + loss_Conditional_discriminator + loss_decoder, epoch * num_iter + x)

            x += 1

        #Save the model after each epoch
        VAE.save("Saved_model\VAE")
        Discriminator.save("Saved_model\Discriminator")
        AAE.save("Saved_model\AAE")
        Encoder.save("Saved_model\Encoder")
        Decoder.save("Saved_model\Decoder")

def test_AAE():
    
    load_Model()

def load_Model():

    Encoder = models.load_model("Saved_model\Encoder")
    Decoder = models.load_model("Saved_model\Decoder")
    VAE = models.load_model("Saved_model\VAE")
    Discriminator = models.load_model("Saved_model\Discriminator")
    AAE = models.load_model("Saved_model\AAE")

    loss_test_VAE = 0
    loss_test_Discriminator = 0
    loss_test_encoder = 0

    print(len(dataset_test))
    i = 0

    for X_test_batch in dataset_test:

        print(i)

        #Phase 1
        loss_test_VAE = ( VAE.test_on_batch(X_test_batch, X_test_batch[0]) + loss_test_VAE) / 2 

        #Phase 2: train the discriminator
        real_distribution = tf.random.normal(shape = [batch_size, coding_size], mean = 0.0, stddev = 5.0)
        _, _, fake_distribution = Encoder(X_test_batch[0])
        X_fake_and_real = tf.concat([fake_distribution, real_distribution], axis = 0)
        label = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
        loss_test_Discriminator = ( Discriminator.test_on_batch(X_fake_and_real, label) + loss_test_Discriminator) /2 

        #Phase 3: train the encoder generator
        label = tf.constant([[1.]] * batch_size)
        loss_test_encoder = ( AAE.test_on_batch(X_test_batch[0], label) + loss_test_encoder) / 2

        i += 1

    print(loss_test_VAE)
    print(loss_test_Discriminator)
    print(loss_test_encoder)

#train_AAE()
test_AAE()