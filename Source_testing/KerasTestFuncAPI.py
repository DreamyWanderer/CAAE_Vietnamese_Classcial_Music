from gc import callbacks
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorboard

from tensorflow import keras
from keras import datasets
from keras import models
from keras import layers
from keras import utils
from keras import callbacks

import pandas as pd
import matplotlib.pyplot as plt

import os

root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%d_%m_%Y-%H_%M_%S")
    
    return os.path.join(root_logdir, run_id)

housing = fetch_california_housing()

X_train_full, X_test, Y_train_full, Y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_full, Y_train_full)

scaler = StandardScaler()
print(X_train)
print('------------------')
X_train = scaler.fit_transform(X_train)
print(X_train)
print(X_valid)
print('------------------')
X_valid = scaler.transform(X_valid)
print(X_valid)
X_test = scaler.transform(X_test)

print(X_train.shape[1:])
input_A = layers.Input(shape = [5], name = "Wide_input")
input_B = layers.Input(shape = [6], name = "deep_input")
hidden_1 = layers.Dense(30, activation = "relu")(input_B)
hidden_2 = layers.Dense(30, activation = "relu")(hidden_1)
concat = layers.Concatenate()([input_A, hidden_2])
output = layers.Dense(1)(concat)
model = models.Model(inputs = [input_A, input_B], outputs = [output])

X_train_A, X_train_B = X_train[:, : 5], X_train[:, 2 :]
X_val_A, X_val_B = X_valid[:, : 5], X_valid[:, 2 :]
X_test_A, X_test_B = X_test[:, : 5], X_test[:, 2 :]

utils.plot_model(model, show_shapes = True, show_dtype = True)
model.summary()

run_logdir = get_run_logdir()
tensorboard_cb = callbacks.TensorBoard(run_logdir)

model.compile(loss = "mean_squared_error", optimizer=keras.optimizers.SGD(lr=0.05))
history = model.fit({"Wide_input": X_train_A, "deep_input": X_train_B}, Y_train, epochs = 20, validation_data = ((X_val_A, X_val_B), Y_valid), callbacks = [tensorboard_cb])
mse_test = model.evaluate((X_test_A, X_test_B), Y_test)

pd.DataFrame(history.history).plot(figsize = (8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

