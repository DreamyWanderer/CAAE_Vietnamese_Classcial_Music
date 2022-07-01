from gc import callbacks
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorboard

import tensorflow as tf
from tensorflow import summary
from tensorflow import keras
from keras import datasets
from keras import models
from keras import layers
from keras import utils
from keras import callbacks

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os

root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%d_%m_%Y-%H_%M_%S")
    
    return os.path.join(root_logdir, run_id)

test_log_dir = get_run_logdir()
writer = summary.create_file_writer(test_log_dir)
with writer.as_default():
    for step in range(1, 1000 + 1):
        print(step)
        images = np.random.rand(1, 32, 32, 3) # random 32×32 RGB images
        tf.summary.image("my_images", images * step / 1000, step=step)