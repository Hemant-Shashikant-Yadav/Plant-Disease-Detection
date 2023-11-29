import keras
from keras import backend as K
from keras.layers.core import Dense, Activation, Dropout
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input, MobileNet
import numpy as np
from IPython.display import Image
from tensorflow.keras.optimizers import Adam

# Load MobileNet with pre-trained weights
base_model = MobileNet(weights='imagenet', include_top=False)

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)  # Add dropout for regularization
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)  # Add dropout for regularization
x = Dense(512, activation='relu')(x)
preds = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)

# Freeze layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,  # Rotate images randomly
    width_shift_range=0.2,  # Shift images horizontally
    height_shift_range=0.2,  # Shift images vertically
    shear_range=0.2,  # Apply shear transformations
    zoom_range=0.2,  # Zoom in randomly
    horizontal_flip=True,  # Flip images horizontally
    fill_mode='nearest'  # Fill in missing pixels using the nearest neighbor
)

# Data augmentation for test data (only preprocessing)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load training and test data
train_generator = train_datagen.flow_from_directory(
    'Y:\Coding\Pycharm ML DL\SIH\\v1.1\dataset\PlantDiseasesDataset\\train',
    target_size=(256, 256),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    'Y:\Coding\Pycharm ML DL\SIH\\v1.1\dataset\PlantDiseasesDataset\\valid',
    target_size=(256, 256),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Train the model
step_size_train = train_generator.n // train_generator.batch_size
model.fit_generator(generator=train_generator, steps_per_epoch=step_size_train, epochs=15)  # Increase epochs

# Evaluate the model on test data
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
