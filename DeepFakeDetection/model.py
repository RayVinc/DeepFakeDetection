import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import models, Sequential, layers, regularizers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def intialize_model():

    image_size = (256, 256)

    model = models.Sequential()

    model.add(layers.Rescaling(1/255, input_shape=(image_size[0],image_size[1],3)))
    model.add(layers.Conv2D(8, kernel_size = (4,4), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Conv2D(16, kernel_size = (3,3), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Conv2D(32, kernel_size = (2,2), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(2, activation='softmax'))

    return model

def compile_model(model):

    metrics = [
        Recall(name='recall'),
        Precision(name='precision'),
        'accuracy']

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=metrics)

def train_model(model, train_ds, val_ds):

    es = EarlyStopping(patience = 10,
                   restore_best_weights = True,
                   monitor = 'val_loss')

    mcp = ModelCheckpoint("{}.h5".format('base_model'),
                      save_weights_only=True,
                      monitor='val_loss',
                      mode='min',
                      verbose=0,
                      save_best_only=True)

    history = model.fit(train_ds,
          validation_data = val_ds,
          batch_size = 32,
          epochs = 50,
          callbacks = [es,mcp])

    print(f"✅ Model trained on train_ds")

    return model, history

'''
make a function for the learning curves
'''

def predict(model, X):

    return model.predict(X)
