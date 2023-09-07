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

def predict(model):

    prediction = ''

    test_easy = image_dataset_from_directory(
        "/kaggle/input/test-data-deepfake-images/real_and_fake_face_test_data/test_easy",
        color_mode = 'rgb',
        shuffle=True,
        batch_size = 32,
        label_mode = "categorical",
        image_size=(256, 256))

    for images, labels in test_easy:
        num_images_to_display = 1

        predictions = model.predict(images)
        predictions = where(predictions < 0.5,0, 1)

        for i in range(num_images_to_display):
            plt.subplot(1, num_images_to_display, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))  # Convert the tensor to a NumPy array
            plt.title(f"Label: {labels[i].numpy()}, pred: {predictions[i]}")
            plt.axis("off")

        plt.show()

    if labels == [1, 0]:
        prediction == 'fake'
    else:
        prediction == 'real'

    return prediction
        # label: [1. 0.] -> fake // label: [0. 1.] -> real
