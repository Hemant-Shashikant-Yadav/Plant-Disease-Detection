import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.models import load_model
from keras import backend as K
import json
import base64

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

img_paths = {
    'A1': 'Y:\Coding\Project\Apple\Plant Disease Detection\\v1.2\Python files\Plant Validation Model\Prediction\img.png',
    'A2': 'Y:\Coding\Project\Apple\Plant Disease Detection\Datasets\Apple dataset\\PlantDiseasesDataset\\train\\Apple___healthy\\a.JPG',
}
img_name_tensors = {name: read_image(img_path) for (name, img_path) in img_paths.items()}

################################################################################################

def top_k_predictions(img, k=2,loaded_model,class_labels):
    image_batch = tf.expand_dims(img, 0)
    predictions = loaded_model(image_batch)
    probs = tf.nn.softmax(predictions, axis=-1)
    top_probs, top_idxs = tf.math.top_k(input=probs, k=k)
    top_labels=class_labels[top_idxs[0][0].numpy()]
    return top_labels, top_probs[0][0]

################################################################################################

PlantValidationsLables = ['It is a plant', 'It is not a plant'] 

PlantValidationModel = load_model(
    "Y:\Coding\Project\Apple\Plant Disease Detection\\v1.2\Python files\Plant Validation Model\PlantValiidationSavedModel.h5",
    custom_objects=custom_objects)

for (name, img_tensor) in img_name_tensors.items():
    pred_label, pred_prob = top_k_predictions(img_tensor,2,PlantValidationModel,PlantValidationsLables)
    print(f'{pred_label}: {pred_prob:0.1%}')

################################################################################################

DiseaseDetectionLables = ['Apple Scab', 'Apple Rot', 'Apple Cedar Rust', 'Apple Healthy']

DiseaseDetectionModel = load_model(
    "Y:\Coding\Project\Apple\Plant Disease Detection\\v1.2\Saved models\SavedModelPlantDetectionMLModel3.h5",
    custom_objects=custom_objects)

for (name, img_tensor) in img_name_tensors.items():
    pred_label, pred_prob = top_k_predictions(img_tensor,2,DiseaseDetectionModel,DiseaseDetectionLables)
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
    image=img_name_tensors['A1'],
    alphas=alphas)




def compute_gradients(images, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = loaded_model(images)
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
    # Generate alphas.
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)

    # Collect gradients.
    gradient_batches = []

    # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]

        gradient_batch = one_batch(baseline, image, alpha_batch, target_class_idx)
        gradient_batches.append(gradient_batch)

    # Concatenate path gradients together row-wise into single tensor.
    total_gradients = tf.concat(gradient_batches, axis=0)

    # Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(gradients=total_gradients)

    # Scale integrated gradients with respect to input.
    integrated_gradients = (image - baseline) * avg_gradients

    return integrated_gradients


@tf.function
def one_batch(baseline, image, alpha_batch, target_class_idx):
    # Generate interpolated inputs between baseline and input.
    interpolated_path_input_batch = interpolate_images(baseline=baseline, image=image, alphas=alpha_batch)

    # Compute gradients between model outputs and interpolated inputs.
    gradient_batch = compute_gradients(images=interpolated_path_input_batch, target_class_idx=target_class_idx)
    return gradient_batch


ig_attributions = integrated_gradients(baseline=baseline, image=img_name_tensors['A1'], target_class_idx=3, m_steps=120)

ig_attributions = integrated_gradients(baseline=baseline, image=img_name_tensors['A2'], target_class_idx=3, m_steps=120)
