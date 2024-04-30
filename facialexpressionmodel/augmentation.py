from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

base_path = "C:/Coding/archive/images/"
batch_size= 128
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=20,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    base_path + "train",
    target_size=(56, 56),  # <-- Specify target_size here
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    base_path + "validation",
    target_size=(56, 56),  # <-- Specify target_size here
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
