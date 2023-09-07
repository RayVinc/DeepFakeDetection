import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

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
    return model

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
          batch_size = 64,
          epochs = 300,
          callbacks = [es,mcp])

    print(f"✅ Model trained on train_ds")

    return model, history

def evaluate(history):

    fig, ax =plt.subplots(1,3,figsize=(20,5))

    # --- LOSS

    ax[0].plot(history.history['loss'])
    ax[0].plot(history.history['val_loss'])
    ax[0].set_title('Model loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend(['Train', 'Val'], loc='upper right')
    ax[0].grid(axis="x",linewidth=0.5)
    ax[0].grid(axis="y",linewidth=0.5)

    # --- RECALL
    ax[1].plot(history.history['recall'])
    ax[1].plot(history.history['val_recall'])
    ax[1].set_title('Model recall', fontsize = 18)
    ax[1].set_xlabel('Epoch', fontsize = 14)
    ax[1].set_ylabel('Recall', fontsize = 14)
    ax[1].legend(['Train', 'Val'], loc='lower right')
    ax[1].grid(axis="x",linewidth=0.5)
    ax[1].grid(axis="y",linewidth=0.5)

    # --- PRECISION

    ax[2].plot(history.history['precision'])
    ax[2].plot(history.history['val_precision'])
    ax[2].set_title('Model precision', fontsize = 18)
    ax[2].set_xlabel('Epoch', fontsize = 14)
    ax[2].set_ylabel('Precision', fontsize = 14)
    ax[2].legend(['Train', 'Val'], loc='lower right')
    ax[2].grid(axis="x",linewidth=0.5)
    ax[2].grid(axis="y",linewidth=0.5)

    plt.show()

    return None


def load_model():

    path_abs = os.getcwd()

    model = tf.keras.models.load_model(os.path.join(path_abs,
                                       'DeepFakeDetection/models/base_model_federico_is_not_helping.h5'),
                                       compile=False)
    return model

def predict(model, image_array):

    image_array = image_array.reshape((1,) + image_array.shape)

    y_pred = model.predict(image_array)[0]
    #y_pred = tf.where(predictions > 0.5,0, 1)
    result = ['fake' if y_pred[0] > 0.5 else 'real' ]

    return {'prob':result[0]}
