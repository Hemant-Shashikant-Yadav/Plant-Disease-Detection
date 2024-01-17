from keras import backend as K
import shap
import numpy as np
import tensorflow as tf

import cv2

print(1)


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


def read_image(image1):
    # image2 = tf.io.decode_jpeg(image1, channels=3)
    image2 = tf.image.convert_image_dtype(image1, tf.float32)
    image3 = tf.image.resize_with_pad(image2, target_height=256, target_width=256)
    return image3


print(2)

custom_objects = {'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m}

from keras.models import load_model

model = load_model('./SavedModelPlantDetectionMLModel3.h5', custom_objects=custom_objects)
print(3)
import os
from sklearn.model_selection import train_test_split

# Path to your local directory containing image data
data_dir = './valid/valid/'

# Get the list of all image files in the directory
scab_images = [os.path.join(data_dir, 'Apple___Apple_scab', img) for img in
               os.listdir(os.path.join(data_dir, 'Apple___Apple_scab'))]
rot_images = [os.path.join(data_dir, 'Apple___Black_rot', img) for img in
              os.listdir(os.path.join(data_dir, 'Apple___Black_rot'))]
rust_images = [os.path.join(data_dir, 'Apple___Cedar_apple_rust', img) for img in
               os.listdir(os.path.join(data_dir, 'Apple___Cedar_apple_rust'))]
healthy_images = [os.path.join(data_dir, 'Apple___healthy', img) for img in
                  os.listdir(os.path.join(data_dir, 'Apple___healthy'))]

# Combine images into a single list
all_images = scab_images + rot_images + rust_images + healthy_images

# Create labels for different classes
scab_labels = [0] * len(scab_images)
rot_labels = [1] * len(rot_images)
rust_labels = [2] * len(rust_images)
healthy_labels = [3] * len(healthy_images)

# Combine labels into a single list
all_labels = scab_labels + rot_labels + rust_labels + healthy_labels
print(3)

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)
print(4)


def convert_and_resize_images(image_paths, target_size=(256, 256)):
    image_arrays = []

    # Iterate through each image path
    for image_path in image_paths:
        # Read the image using OpenCV
        image = cv2.imread(image_path)

        # Convert the image to RGB format (OpenCV reads images in BGR format)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image to the target size
        resized_image = cv2.resize(image_rgb, target_size)

        # Convert the resized image to a NumPy array
        image_array = np.array(resized_image)

        # Append the NumPy array to the list
        image_arrays.append(image_array)

    # Convert the list of NumPy arrays to a NumPy array
    image_arrays1 = np.array(image_arrays)

    return image_arrays1


print(5)

xtrain_arrays = convert_and_resize_images(xtrain, target_size=(256, 256))
print(6)
xtrain_arrays_float32 = xtrain_arrays.astype('float32') / 255.0
print(7)
e = shap.DeepExplainer(model, xtrain_arrays_float32)
print(8)
