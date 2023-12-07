from keras.applications.mobilenet import preprocess_input
from keras.models import load_model
from keras import backend as K
from matplotlib import pyplot as plt


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

# Load the saved model
# loaded_model = load_model("Y:\Coding\Project\Apple\Plant Disease Detection\\v1.2\\tempMobileNet Transfer Learning111.h5")
loaded_model = load_model(
    "Y:\Coding\Project\Apple\Plant Disease Detection\\v1.2\\tempMobileNet Transfer Learning111.h5",
    custom_objects=custom_objects)

from keras.preprocessing import image
import numpy as np

# Load and preprocess the image to be predicted
img_path = 'Y:\Coding\Project\Apple\Plant Disease Detection\Datasets\Full dataset\\test\\test\AppleCedarRust1.JPG'  # Replace with the path to your image
# img = image.load_img(img_path, target_size=(256, 256))
img = image.load_img(img_path)

img = image.img_to_array(img)
# print(img.shape)
img = np.expand_dims(img, axis=0)
# print(img.shape)
img = preprocess_input(img)
# print(img.shape)

# Make predictions
predictions = loaded_model.predict(img)

# Get the predicted class
predicted_class_index = np.argmax(predictions)

class_labels = ['apple scab', 'apple rot', 'apple cedar rust', 'apple healthy']  # Replace with your actual class labels
predicted_class = class_labels[predicted_class_index]
print(f"The predicted class is: {predicted_class}")
