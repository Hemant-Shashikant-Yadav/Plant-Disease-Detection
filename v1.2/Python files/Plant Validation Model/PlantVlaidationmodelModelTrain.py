# testing
import pandas as pd
from keras import backend as K
from tensorflow.keras.layers import Dense, Activation, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.applications.mobilenet import preprocess_input, MobileNet
from tensorflow.keras.optimizers import Adam

base_model = MobileNet(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
preds = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)

for layer in base_model.layers:
    layer.trainable = False


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


# Compile the model with custom metrics
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    'Y:\Coding\Project\Apple\Plant Disease Detection\Datasets\Dataset1\\train_11',
    target_size=(256, 256),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    'Y:\Coding\Project\Apple\Plant Disease Detection\Datasets\Dataset1\\validate_11',
    target_size=(256, 256),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

step_size_train = train_generator.n // train_generator.batch_size

his = model.fit(train_generator, steps_per_epoch=step_size_train, epochs=15, validation_data=test_generator)

# Saving the training history to a JSON file
hist_df = pd.DataFrame(his.history)
hist_json_file = 'history.json'
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# Saving the model
model.save("PlantValiidationSavedModel.h5")

# Evaluating the model
result = model.evaluate(test_generator)
loss = result[0]
accuracy = result[1]
f1_score = result[2]
precision = result[3]
recall = result[4]

print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
print(f"F1 score: {f1_score}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
