import os
import tempfile
import json
import base64
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.models import load_model
from keras import backend as K


################################################################################################

def read_image(file_name):
    image = tf.io.read_file(file_name)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_pad(image, target_height=256, target_width=256)
    return image


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

################################################################################################

img_path = 'Y:\Coding\Project\Apple\Plant Disease Detection\v1.2\Python files\Prediction models\car.jpg'
img_name_tensors = read_image(img_path)


################################################################################################

def top_k_predictions(img, k, loaded_model, class_labels):
    image_batch = tf.expand_dims(img, 0)
    predictions = loaded_model(image_batch)
    probs = tf.nn.softmax(predictions, axis=-1)
    top_probs, top_idxs = tf.math.top_k(input=probs, k=k)
    top_labels = class_labels[top_idxs[0][0].numpy()]
    return top_labels, top_probs[0][0]


################################################################################################

PlantValidationsLables = ['It is a plant', 'It is not a plant']

PlantValidationModel = load_model(
    "Y:\Coding\Project\Apple\Plant Disease Detection\\v1.2\Python files\Plant Validation Model\PlantValiidationSavedModel.h5",
    custom_objects=custom_objects)

pred_label, pred_prob = top_k_predictions(img_name_tensors, 2, PlantValidationModel, PlantValidationsLables)
print(f'{pred_label}: {pred_prob:0.1%}')

################################################################################################

DiseaseDetectionLables = ['Apple Scab', 'Apple Rot', 'Apple Cedar Rust', 'Apple Healthy']

DiseaseDetectionModel = load_model(
    "Y:\Coding\Project\Apple\Plant Disease Detection\\v1.2\Saved models\SavedModelPlantDetectionMLModel3.h5",
    custom_objects=custom_objects)

pred_label, pred_prob = top_k_predictions(img_name_tensors, 3, DiseaseDetectionModel, DiseaseDetectionLables)
print(f'{pred_label}: {pred_prob:0.1%}')

################################################################################################

baseline = tf.zeros(shape=(256, 256, 3))

m_steps = 20
alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)


def interpolate_images(baseline, image, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x + alphas_x * delta
    return images


interpolated_images = interpolate_images(
    baseline=baseline,
    image=img_name_tensors,
    alphas=alphas)


def compute_gradients(images, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = DiseaseDetectionModel(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, images)


path_gradients = compute_gradients(
    images=interpolated_images,
    target_class_idx=3)


def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients


ig = integral_approximation(
    gradients=path_gradients)


def integrated_gradients(baseline, image, target_class_idx, m_steps=30, batch_size=8):
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)
    gradient_batches = []

    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]
