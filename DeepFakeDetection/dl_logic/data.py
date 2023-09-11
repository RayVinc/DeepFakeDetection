import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image, ImageChops, ImageEnhance
import os

from tensorflow.keras.utils import image_dataset_from_directory

# Update this part with the latest model / directory.
def loading_data():

    image_size = (256, 256)

    train_ds, val_ds = image_dataset_from_directory(
        "/kaggle/input/deepfakedetection-lw/Project_Data/real_and_fake_face",
        validation_split=0.2,
        color_mode = 'rgb',
        subset="both",
        label_mode = "categorical",
        seed=123,
        batch_size = 32,
        image_size=image_size)

    return train_ds, val_ds
