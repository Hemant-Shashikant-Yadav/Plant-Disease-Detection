import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def setDataset(name):
    print(f"Y:\Coding\Pycharm ML DL\SIH\\v1.1\dataset\PlantDiseasesDataset\\{name}")
    dataset = image_dataset_from_directory(
        f"Y:\Coding\Pycharm ML DL\SIH\\v1.1\dataset\PlantDiseasesDataset\\{name}",
        shuffle=True,
        image_size=(256, 256),
        batch_size=32
    )
    return dataset


trainDataset = setDataset("train")
validDataset = setDataset("valid")

testDataset = image_dataset_from_directory(
    f"Y:\Coding\Pycharm ML DL\SIH\\v1.1\dataset\\test",
    shuffle=True,
    image_size=(256, 256),
    batch_size=32
)

trainDataset = trainDataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
validDataset = validDataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
testDataset = testDataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

resize_rescale = tf.keras.Sequential([

    layers.experimental.preprocessing.Resizing(256, 256),
    layers.experimental.preprocessing.Rescaling(1.0 / 255),
])

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.3),
])

trainDataset = trainDataset.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flatten layer to transition from convolutional to fully connected layers
    layers.Flatten(),

    # Fully connected layers
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Dropout for regularization
    layers.Dense(64, activation='relu'),

    # Output layer with softmax activation for classification
    layers.Dense(4, activation='softmax')
])
model.build((32, 256, 256, 3))

print(model.summary())

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = model.fit(
    trainDataset,
    batch_size=32,
    validation_data=validDataset,
    verbose=1,
    epochs=20,
    callbacks=[early_stopping, reduce_lr],
)

scores = model.evaluate(testDataset)

print(scores)
