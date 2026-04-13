import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(path):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True
    )

    data = datagen.flow_from_directory(
        path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )
    return data