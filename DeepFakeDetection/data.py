import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.utils import image_dataset_from_directory

def loading_data():

    image_size = (256, 256)

    train_ds, val_ds = image_dataset_from_directory(
        "/kaggle/input/real-and-fake-face-detection/real_and_fake_face", #FIX IT!!! We need to have the data localy on the RAW DATA folder
        validation_split=0.2,
        subset="both",
        label_mode = "categorical",
        seed=123,
        batch_size = 32,
        image_size=image_size)

    return train_ds, val_ds
