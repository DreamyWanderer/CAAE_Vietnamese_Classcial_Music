from calendar import day_abbr
from turtle import back
from muspy import load_json
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
import Preprocess

curmin = -127
curmax = 210
label_normalized = {}
metadata_pianoroll = Preprocess.openJSONFile("Dataset\Samples\link.json")

idPianoroll = 0

def loadJSONNormalized(path):

    for root, dirs, files in os.walk(path):
        for filename in files:

            num, ext = os.path.splitext(filename)
            if ext == '.json':
                label_normalized[num] = Preprocess.openJSONFile(os.path.join(root, filename))

def scale(tensorPianoroll, reverse = False):

    if not reverse:
        return (tensorPianoroll -  curmin) / (curmax - curmin)
    else:
        return tensorPianoroll * (curmax - curmin) + curmin

'''def loadPianoroll(filepath):

    filepath = bytes.decode(filepath)
    idPianoroll = os.path.basename(os.path.splitext(filepath)[0])
    pianoroll = np.load(filepath)
    tensor_pianoroll = tf.convert_to_tensor(pianoroll, dtype = tf.float16)
    label = Config.small_cag[ Config.big_cag[ label_normalized[ metadata_pianoroll[idPianoroll] ]["style"] ] ][ label_normalized[ metadata_pianoroll[idPianoroll] ]["emotion"] ]
    one_hot = tf.one_hot(label, depth = 8)

    return tf.data.Dataset.from_tensors(tensor_pianoroll, one_hot)'''

def pianoroll_reader_dataset(filepaths, batch_size = Config.batch_size):

    dataset = tf.data.Dataset.list_files(filepaths, shuffle = False)
    dataset = dataset.map(lambda filepath: tf.numpy_function(loadPianoroll, [filepath], [tf.float16]), num_parallel_calls = tf.data.AUTOTUNE)
    dataset = dataset.map(scale, num_parallel_calls = tf.data.AUTOTUNE)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def loadPianoroll(filepath, num = 100):

    global idPianoroll

    #Init for shape
    data = (np.load( os.path.join(filepath, str(idPianoroll) + ".npy") ), )
    label = ()

    for i in range(2, num + 1):

        pianoroll = (np.load( os.path.join(filepath, str(idPianoroll) + ".npy") ), )
        if pianoroll[0].shape == data[0].shape:
            data += pianoroll
        else:
            break
        #one_hot = tf.one_hot(label, depth = 8)

        idPianoroll += 1

    data = np.stack(data, axis = 0)
    label = np.stack(label, axis = 0)
    label = utils.to_categorical(label, 8)

    return tf.data.Dataset.from_tensors(data, label)

loadJSONNormalized("Dataset\Dataset_normalized")
loadPianoroll("Dataset\Samples\\")
