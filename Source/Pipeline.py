import tensorflow as tf
from tensorflow import summary
from tensorflow import keras
from tensorflow import data
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
import re

import Config
import Preprocess

curmin = -127
curmax = 210
metadata_normalized = {}
metadata_pianoroll = Preprocess.openJSONFile("Dataset\Samples\link.json")
bucket_samples = {} #Store and determine the bucket for samples with different length. The key is number of timestep stored in pianoroll, value is id of bucket to put pinaroll into.
bucket_bound = [x for x in range (1, 900)]
bucket_size = [Config.batch_size] * 900

idPianoroll = 0

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def getLabel(fileName):
    '''
    Return the number describe the label of the pianoroll, prepare for converting to one-hot vector
    '''
    num, ext = os.path.splitext(fileName)
    normalized = metadata_normalized[ metadata_pianoroll[num] ]
    bigCag = Config.big_cag[ normalized["style"] ]
    smallCag = Config.small_cag[ bigCag ][ normalized["emotion"] ]

    return smallCag

def loadJSONNormalized(path):

    for root, dirs, files in os.walk(path):
        for filename in files:

            num, ext = os.path.splitext(filename)
            if ext == '.json':
                metadata_normalized[num] = Preprocess.openJSONFile(os.path.join(root, filename))

def loadFileGenerator():
    '''
    Load the pianoroll samples from hard disk drive and transfering them to the dataset. This function is a generator.
    '''
    path = "Dataset/Samples/Pianorolls"

    for root, dirs, files in os.walk(path):
        for filename in sorted(files, key = natural_keys):
            print(filename)

            sample = np.load(os.path.join(path, filename) )
            oneHot = utils.to_categorical( getLabel(filename), 8, dtype = "uint8" )

            yield sample, oneHot

def lengthSample(pianoroll):
    '''
    Find the length of the sample, so that we can put it in the suitable bucket using for further batching process
    '''
    global bucket_samples

    l = int(tf.shape( pianoroll )[0])

    if l not in bucket_samples:
        print("hey")
        bucket_samples[l] = len(bucket_samples)

    return bucket_samples[l]

#In case need using lots of tracing protocol, use the function signatures instead
@tf.function
def MinMaxScalerTransform(array, min, max, range):

    return tf.divide(array - min, max - min) * (range[1] - range[0]) + range[0]

@tf.function
def normalizedPianoroll(pianoroll, onehot):
    '''
    Scale the pinaoroll value to range [-1, 1]
    '''

    print("Tracing ________")

    melody = tf.cast( MinMaxScalerTransform(pianoroll[::, ::, 0 : 128], -127, 127, (-1, 1) ), tf.float16)
    beat = tf.cast(pianoroll[::, ::, 128 : 129], tf.float16)
    tempo = tf.cast( MinMaxScalerTransform(pianoroll[::, ::, 129 :], 0, 210, (-1, 1) ), tf.float16)

    return tf.concat([melody, beat, tempo], 2), onehot

loadJSONNormalized("Dataset\Dataset_normalized")

#Load pianorolls from disk
dataset = tf.data.Dataset.from_generator(loadFileGenerator, output_signature = (tf.TensorSpec(shape = (None, 130), dtype = tf.int16), tf.TensorSpec(shape = (8, ), dtype = tf.uint8) ) )

#Create batches divided by length of the pianorolls
dataset = dataset.bucket_by_sequence_length(lambda pianoroll, onehot: tf.py_function(lengthSample, inp = [pianoroll], Tout = tf.int32), bucket_boundaries = bucket_bound, bucket_batch_sizes = bucket_size)

#Normalized pianorolls value to [-1, 1] range
dataset = dataset.map(lambda pianoroll, onehot: normalizedPianoroll(pianoroll, onehot), num_parallel_calls = tf.data.AUTOTUNE)

#Cache dataset
dataset = dataset.cache("Dataset\\Cached\\test")

#Prefetch dataset
dataset = dataset.prefetch(tf.data.AUTOTUNE)

for data in dataset:
    print(data)
