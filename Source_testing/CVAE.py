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

coding_size = 32
num_type = 10
batch_size = 32
num_epoch = 20

root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%d_%m_%Y-%H_%M_%S")
    
    return os.path.join(root_logdir, run_id)

class Sampling(layers.Layer):
    
    def call(self, inputs):
        mean, log_var = inputs
        
        return backend.random_normal(tf.shape(log_var)) * backend.exp(log_var/2) + mean

def plot_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")

def show_reconstructions(model, n_images=10):
    reconstructions = model.predict( (X_valid[:n_images], np.tile(one_hot_train[3], (n_images, 1)) ))
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plot_image(X_valid[image_index])
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plot_image(reconstructions[image_index])
    plt.show()

fashion_mnist = datasets.fashion_mnist
(X_train_full, Y_train_full), (X_test, Y_test) = fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
Y_valid, Y_train = Y_train_full[:5000], Y_train_full[5000:]
class_name = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
one_hot_train = utils.to_categorical(Y_train, num_classes = num_type)
one_hot_valid = utils.to_categorical(Y_valid, num_classes = num_type)

X_train = np.float32(X_train)
dataset = tf.data.Dataset.from_tensor_slices( (X_train, one_hot_train) )
dataset = dataset.batch(batch_size, drop_remainder = True).prefetch(1)

#Define Encoder model
def construct_Encoder():

    input_encoder = layers.Input( (28, 28), name = "Encoder_input")
    flatten_1 = layers.Flatten( name = "Flatten_layer")(input_encoder)
    hidden_encoder_1 = layers.Dense(150, activation = "selu", name = "Hidden_layer_1")(flatten_1)
    hidden_encoder_2 = layers.Dense(100, activation = "selu", name = "Hidden_layer_2")(hidden_encoder_1)
    coding_mean = layers.Dense(coding_size, name = "Mean_layer")(hidden_encoder_2)
    coding_log_var = layers.Dense(coding_size, name = "Variance_layer")(hidden_encoder_2)
    output_encoder = Sampling(name = "Encoder_output")([coding_mean, coding_log_var])
    Encoder = models.Model(input_encoder, outputs = [coding_mean, coding_log_var, output_encoder], name = "Encoder")

    utils.plot_model(Encoder, show_shapes = True, show_dtype = True, to_file = "Encoder.png")

    return Encoder

#Define Decoder model
def construct_Decoder():

    input_decoder = layers.Input( (coding_size + num_type, ), name = "Decoder_input")
    hidden_decoder_1 = layers.Dense(100, activation = "selu", name = "Hidden_layer_1_d")(input_decoder)
    hidden_decoder_2 = layers.Dense(150, activation = "selu", name = "Hidden_layer_2_d")(hidden_decoder_1)
    output_decoder = layers.Dense(28 * 28, activation = "sigmoid", name = "Decoder_output")(hidden_decoder_2)
    reshape_1 = layers.Reshape( (28, 28), name = "Reshape_layer" )(output_decoder)
    Decoder = models.Model(input_decoder, reshape_1, name = "Decoder")

    utils.plot_model(Decoder, to_file= "Decoder.png", show_shapes = True, show_dtype = True)

    return Decoder

#Define Variational VAE model
def construct_VAE(usingKL = True):

    Encoder = construct_Encoder()
    Decoder = construct_Decoder()

    VAE_input = Encoder.input
    Label_input = layers.Input( (num_type, ), name = "Lable_input")
    _, _, latent_space = Encoder(VAE_input)
    concat_layer = layers.Concatenate(name = "Concatenate_layer")([latent_space, Label_input])
    VAE_output = Decoder(concat_layer)
    VAE = models.Model(inputs = [VAE_input, Label_input], outputs = VAE_output, name = "VAE")
    utils.plot_model(VAE, to_file= "VAE.png", show_shapes = True, show_dtype = True)

    coding_log_var = Encoder.get_layer("Variance_layer").output
    coding_mean = Encoder.get_layer("Mean_layer").output
    latent_loss = -0.5 * backend.sum( 1 + coding_log_var - backend.exp(coding_log_var) - backend.square(coding_mean), axis = -1 )
    if usingKL:
        VAE.add_loss( backend.mean(latent_loss) / 784 )
    VAE.compile(loss = "binary_crossentropy", optimizer = optimizers.Adam(learning_rate = 0.0002) )

    return VAE

# stuff only to run when not called via 'import' here
if __name__ == "__main__":

    VAE = construct_VAE()

    #Train VAE
    history = VAE.fit( [X_train, one_hot_train], X_train, epochs = 20, validation_data = [ [X_valid, one_hot_valid], X_valid])

    #Show history data
    pd.DataFrame(history.history).plot(figsize = (8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

    #Show reconstruction image result
    show_reconstructions(VAE)

    #Latent space
    _, _, X_valid_compressed = VAE.get_layer("Encoder")(X_valid)
    tsne = TSNE()
    X_valid_2D = tsne.fit_transform(X_valid_compressed)
    plt.scatter(X_valid_2D[:, 0], X_valid_2D[:, 1], c=Y_valid, s=10, cmap="tab10")
    plt.show()