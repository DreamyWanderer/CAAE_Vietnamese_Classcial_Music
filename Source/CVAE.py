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
import os

import Config

def get_run_logdir():
    import time
    run_id = time.strftime("run_%d_%m_%Y-%H_%M_%S")
    
    return os.path.join(Config.root_logdir, run_id)

class Sampling(layers.Layer):
    
    def call(self, inputs):

        mean, log_var = inputs
        
        return backend.random_normal(tf.shape(log_var)) * backend.exp(log_var/2) + mean

class RepeatTime(layers.Layer):

    def call(self, inputs):
        
        vector, timestep = inputs
        numTimestep = tf.shape(timestep)[1]

        return backend.repeat(vector, numTimestep)

def metric_accuracy(y_true, y_pred):
    '''
    Calculate accuracy of reconstruction
    '''

    first_part_compare = backend.equal(tf.math.ceil(y_true[::, ::, 0:129]), tf.math.ceil(y_pred[::, ::, 0:129]) ) #Construction accuracy of melody and beat part
    second_part_compare = backend.equal( y_true[::, ::, 129:], y_pred[::, ::, 129:] )
    combine_part = tf.concat( [first_part_compare, second_part_compare], -1)

    return backend.mean(combine_part)

#Define Encoder model
def construct_Encoder():

    depth = Config.encoder_param["depth"]

    input_encoder = layers.Input( (None, Config.num_feature), name = "Encoder_input")

    x = input_encoder
    for i in range(depth - 1):
        x = layers.LSTM(Config.encoder_param["num_hidden_node"], return_sequences = True, name = f"LSTM_encoder_{i + 1}")(x)

    end_LSTM = layers.LSTM(Config.encoder_param["num_hidden_node"], name = f"LSTM_encoder_{depth}")(x)
    coding_mean = layers.Dense(Config.coding_size, name = "Mean_layer")(end_LSTM)
    coding_log_var = layers.Dense(Config.coding_size, name = "Variance_layer")(end_LSTM)
    output_encoder = Sampling(name = "Encoder_output")([coding_mean, coding_log_var])
    Encoder = models.Model(input_encoder, outputs = [coding_mean, coding_log_var, output_encoder], name = "Encoder")

    utils.plot_model(Encoder, show_shapes = True, show_dtype = True, to_file = "Document\\Model_diagram\\Encoder.png")

    return Encoder

#Define Decoder model
def construct_Decoder():

    input_decoder = layers.Input( (Config.coding_size + Config.num_type, ), name = "Decoder_input")
    input_timestep = layers.Input( (None, Config.num_feature), name = "Decoder_input_num_timestep")
    repeat_layer = RepeatTime()([input_decoder, input_timestep])
    x = repeat_layer

    depth = Config.decoder_param["depth"]
    for i in range(depth):
        x = layers.LSTM(Config.decoder_param["num_hidden_node"], return_sequences = True, name = f"LSTM_decoder_{i + 1}")(x)
    
    output_decoder = layers.TimeDistributed( layers.Dense(Config.num_feature, activation = "tanh", name = "Decoder_output"))(x)

    Decoder = models.Model([input_decoder, input_timestep], output_decoder, name = "Decoder")

    utils.plot_model(Decoder, to_file= "Document\\Model_diagram\\Decoder.png", show_shapes = True, show_dtype = True)

    return Decoder

#Define Variational VAE model
def construct_VAE(usingKL = True):

    Encoder = construct_Encoder()
    Decoder = construct_Decoder()

    VAE_input = Encoder.input
    Label_input = layers.Input( (Config.num_type, ), name = "Lable_input")
    _, _, latent_space = Encoder(VAE_input)
    concat_layer = layers.Concatenate(name = "Concatenate_layer")([latent_space, Label_input])
    VAE_output = Decoder([concat_layer, VAE_input])
    VAE = models.Model(inputs = [VAE_input, Label_input], outputs = VAE_output, name = "VAE")
    utils.plot_model(VAE, to_file= "Document\\Model_diagram\\VAE.png", show_shapes = True, show_dtype = True)

    coding_log_var = Encoder.get_layer("Variance_layer").output
    coding_mean = Encoder.get_layer("Mean_layer").output
    latent_loss = -0.5 * backend.sum( 1 + coding_log_var - backend.exp(coding_log_var) - backend.square(coding_mean), axis = -1 )
    if usingKL:
        VAE.add_loss( backend.mean(latent_loss) / 784 )

    VAE.compile(loss = "binary_crossentropy", optimizer = optimizers.Adam(learning_rate = 0.0002), metrics = [metric_accuracy] )

    return VAE

# stuff only to run when not called via 'import' here
if __name__ == "__main__":

    VAE = construct_VAE()

    #Train VAE

    #Show history data
    '''pd.DataFrame(history.history).plot(figsize = (8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()'''

    #Latent space
    '''_, _, X_valid_compressed = VAE.get_layer("Encoder")(X_valid)
    tsne = TSNE()
    X_valid_2D = tsne.fit_transform(X_valid_compressed)
    plt.scatter(X_valid_2D[:, 0], X_valid_2D[:, 1], c=Y_valid, s=10, cmap="tab10")
    plt.show()'''