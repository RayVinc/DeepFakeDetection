import matplotlib.pyplot as plt
import os
import tensorflow as tf

from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ReduceLROnPlateau , ModelCheckpoint, CSVLogger

#Best pratice - not used outside Kaggle
def intialize_compile_model():
    densenet = ResNet50(
                        weights='imagenet',
                        include_top=False,
                        input_shape=(128,128,3)
                        )
    model = Sequential([densenet,
                        layers.GlobalAveragePooling2D(),
                        layers.Dense(512,activation='relu'),
                        layers.BatchNormalization(),
                        layers.Dense(2, activation='softmax')
                        ])
    model.compile(optimizer=Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                 )
    return model

#Best pratice - not used outside Kaggle
def train_model(model, train_flow, valid_flow):
    es = callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    checkpoint = ModelCheckpoint(filepath='model_20epochs.h5',
                                save_best_only=True,
                                save_weights_only=False,
                                verbose=1,
                                mode='min',
                                monitor='val_loss'
                                )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                factor=0.2,
                                patience=3,
                                verbose=1,
                                min_delta=0.0001
                                )

    csv_logger = CSVLogger('training.log')

    callbacks = [checkpoint, reduce_lr, csv_logger, es]

    train_steps = 100000//64
    valid_steps = 20000//64

    history = model.fit(train_flow,
        epochs = 30,
        steps_per_epoch =train_steps,
        validation_data =valid_flow,
        validation_steps = valid_steps,
        callbacks=callbacks
    )

    return model, history

#Best pratice - not used outside Kaggle
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

# Use to load the model
def load_model():

    path_abs = os.getcwd()

    model = tf.keras.models.load_model(os.path.join(path_abs,
                                        'DeepFakeDetection/models/model_20epochs.h5'),
                                        compile=False)
    return model

# Use to make a prediction on a new image and send back probabiliy
def predict(model, image_array):

    y_pred = model.predict(image_array)
    #print(f'{y_pred = }')

    return {'prob': y_pred.tolist()[0]}
