from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input
import shap
import numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split


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


custom_objects = {'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m}

model = load_model('./SavedModelPlantDetectionMLModel3.h5', custom_objects=custom_objects)

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

data_generator = datagen.flow_from_directory(
    'E:\Coding\Project\Apple\Plant Disease Detection\\v1.4 XAI\\valid\\valid',
    target_size=(128, 128),
    color_mode='rgb',
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

data = []
labels = []
for _ in range(len(data_generator)):
    batch_data, batch_labels = data_generator.next()
    data.extend(batch_data)
    labels.extend(batch_labels)

# Convert the lists to arrays
data = np.array(data)
labels = np.array(labels)

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=0.2, random_state=42)

print(len(xtrain))
print(len(xtest))
print(type(xtest))
print(type(xtest))
e = shap.DeepExplainer(model, xtrain)
