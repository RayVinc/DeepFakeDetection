import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image, ImageChops, ImageEnhance
import os

from tensorflow.keras.utils import image_dataset_from_directory

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

def ela(input_image, quality):
    """
    Generates an ELA image from input image
    """
    tmp_fname = "temp_image"
    ela_fname = "result_image.jpeg"

    im = Image.fromarray(input_image)
    im = im.convert('RGB')
    im.save(tmp_fname, 'JPEG', quality=quality)

    tmp_fname_im = Image.open(tmp_fname)
    ela_im = ImageChops.difference(im, tmp_fname_im)

    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0/max_diff
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

    ela_im.save(ela_fname, 'JPEG')
    os.remove(tmp_fname)
    return ela_fname
