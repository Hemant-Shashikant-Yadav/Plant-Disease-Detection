from keras import backend as K
from keras.layers.core import Dense, Activation, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.mobilenet import preprocess_input, MobileNet
from tensorflow.keras.optimizers import Adam
import json

from tensorflow.python.keras.callbacks import History

# Load MobileNet with pre-trained weights
base_model = MobileNet(weights='imagenet', include_top=False)

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)  # Add dropout for regularization
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)  # Add dropout for regularization
x = Dense(128, activation='relu')(x)
preds = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)

# Freeze layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
optimizer = Adam(learning_rate=0.001)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])

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
    'Y:\Coding\Jupyter\SIH\\v1.1\dataset\PlantDiseasesDataset\\train',
    target_size=(256, 256),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    'Y:\Coding\Jupyter\SIH\\v1.1\dataset\PlantDiseasesDataset\\valid',
    target_size=(256, 256),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

history_callback = History()

# Train the model
step_size_train = train_generator.n // train_generator.batch_size
model.fit_generator(generator=train_generator, steps_per_epoch=step_size_train, epochs=12, callbacks=[history_callback])  # Increase epochs

history = history_callback.history

# Save the history to a JSON file
with open('training_history.json', 'w') as json_file:
    json.dump(history, json_file)

model.save("tempMobileNet Transfer Learning111.h5")
# Evaluate the model on test data
loss, accuracy, f1_m, precision_m, recall_m = model.evaluate(test_generator)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
print(f"F1 score: {f1_m}")
print(f"Precision: {precision_m}")
print(f"Recall: {recall_m}")
