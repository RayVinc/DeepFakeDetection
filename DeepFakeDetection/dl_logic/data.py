from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Best pratice - not used outside Kaggle
def loading_data():

    base_path = '/kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/'

    ig = ImageDataGenerator(rescale=1./255.)
    train_flow = ig.flow_from_directory(
        base_path + 'train/',
        target_size=(128, 128),
        batch_size=64,
        class_mode='categorical'
    )

    valid_flow = ig.flow_from_directory(
        base_path + 'valid/',
        target_size=(128, 128),
        batch_size=64,
        class_mode='categorical'
    )

    test_flow = ig.flow_from_directory(
        base_path + 'test/',
        target_size=(128, 128),
        batch_size=1,
        shuffle = False,
        class_mode='categorical'
)
    return train_flow, valid_flow, test_flow
