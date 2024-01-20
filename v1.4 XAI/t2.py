from keras.applications.vgg16 import preprocess_input
import json
import shap
import keras
from keras import backend as K

from keras.models import load_model


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

loaded_model = load_model(
    './SavedModelPlantDetectionMLModel3.h5',
    custom_objects=custom_objects)

from PIL import Image
import numpy as np
import os


def read_images_from_directory(directory_path):
    image_arrays = []

    # Iterate through subdirectories in the main directory
    for subdirectory in os.listdir(directory_path):
        subdirectory_path = os.path.join(directory_path, subdirectory)

        # Check if the subdirectory is a directory
        if os.path.isdir(subdirectory_path):
            # Iterate through images in the subdirectory
            for filename in os.listdir(subdirectory_path):
                image_path = os.path.join(subdirectory_path, filename)

                # Open image using PIL
                image = Image.open(image_path)

                # Convert PIL Image to numpy array
                image_array = np.array(image)

                # Append the numpy array to the list
                image_arrays.append(image_array)

    return image_arrays


# Example usage
directory_path = 'E:\Coding\Project\Apple\Plant Disease Detection\\v1.4 XAI\\valid\\valid'
all_image_arrays = read_images_from_directory(directory_path)

# Now, all_image_arrays contains numpy arrays of all the images in the specified directory structure
arr = np.asarray(all_image_arrays)
arr = arr.astype('float32')
arr /= 255
bf = arr[:100]
e = shap.DeepExplainer(loaded_model, bf)

input_sample = arr[200]
print(input_sample.shape)
# Ensure that the input_sample has the correct shape for your model
shap_values = e.shap_values(input_sample_reshaped)

# Plot the SHAP values
shap.image_plot(shap_values, -input_sample_reshaped)
