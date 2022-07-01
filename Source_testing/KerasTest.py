import pydot
import graphviz
import tensorflow as tf
from tensorflow import keras
from keras import datasets
from keras import models
from keras import layers
from keras import utils

import pandas as pd
import matplotlib.pyplot as plt

#Sequenticail API

print(tf.__version__)
print(keras.__version__)

fashion_mnist = datasets.fashion_mnist
(X_train_full, Y_train_full), (X_test, Y_test) = fashion_mnist.load_data()
print(X_train_full.shape)
print(X_train_full.dtype)

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
Y_valid, Y_train = Y_train_full[:5000], Y_train_full[5000:]
class_name = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(Y_train)
print(class_name[Y_train[0]])

model = models.Sequential()
model.add(layers.Flatten(input_shape = [28, 28]))
model.add(layers.Dense(300, activation = "relu"))
model.add(layers.Dense(100, activation = "relu"))
model.add(layers.Dense(10, activation = "softmax"))

utils.plot_model(model)

model.compile(loss = "sparse_categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])

history = model.fit(X_train, Y_train, epochs = 30, validation_data = (X_valid, Y_valid))

pd.DataFrame(history.history).plot(figsize = (8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

X_new = X_test[:3]
Y_pro = model.predict(X_new)
print(Y_pro)
