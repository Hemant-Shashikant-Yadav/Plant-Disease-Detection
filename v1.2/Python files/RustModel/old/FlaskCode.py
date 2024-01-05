import base64

from flask import Flask, render_template, request, jsonify
from io import BytesIO
import tensorflow as tf
from PIL import Image
import numpy as np
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.models import load_model
from keras import backend as K
from matplotlib import pyplot as plt

app = Flask(__name__)


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


def top_k_predictions(img, k, loaded_model, class_labels):
    image_batch = tf.expand_dims(img, 0)
    predictions = loaded_model(image_batch)
    probs = tf.nn.softmax(predictions, axis=-1)
    top_probs, top_idxs = tf.math.top_k(input=probs, k=k)
    top_labels = class_labels[top_idxs[0][0].numpy()]
    return top_labels


def interpolate_images(baseline, image, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x + alphas_x * delta
    return images


def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients


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


def plot_img_attributions_and_save(baseline, image, target_class_idx, m_steps=20, cmap=None, overlay_alpha=0.4):
    attributions = integrated_gradients(baseline=baseline, image=image, target_class_idx=target_class_idx,
                                        m_steps=m_steps)

    attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

    original_image_base64 = base64.b64encode(image.numpy()).decode('utf-8')
    overlay_image_base64 = base64.b64encode(attribution_mask.numpy()).decode('utf-8')

    data = {
        'overlay_image': overlay_image_base64,
        'alpha': overlay_alpha,
    }

    return data


def process_image(temp_image_path):
    img_name_tensors = read_image(temp_image_path)

    PlantValidationsLables = ['It is a plant', 'It is not a plant']

    PlantValidationModel = load_model(
        "/v1.2/Python files/Plant Validation Model/PlantValiidationSavedModel.h5",
        custom_objects=custom_objects)

    pred_label = top_k_predictions(img_name_tensors, 2, PlantValidationModel, PlantValidationsLables)

    if pred_label == 'It is not a plant':
        return "This is not plant image", temp_image_path

    DiseaseDetectionLables = ['Apple Scab', 'Apple Rot', 'Apple Cedar Rust', 'Apple Healthy']

    global DiseaseDetectionModel
    DiseaseDetectionModel = load_model(
        "/v1.2/Saved models/SavedModelPlantDetectionMLModel3.h5",
        custom_objects=custom_objects)

    pred_label1 = top_k_predictions(img_name_tensors, 3, DiseaseDetectionModel, DiseaseDetectionLables)

    baseline = tf.zeros(shape=(256, 256, 3))

    m_steps = 20
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)

    interpolated_images = interpolate_images(
        baseline=baseline,
        image=img_name_tensors,
        alphas=alphas)

    path_gradients = compute_gradients(
        images=interpolated_images,
        target_class_idx=3)

    ig = integral_approximation(
        gradients=path_gradients)

    ig_attributions = integrated_gradients(baseline=baseline, image=img_name_tensors, target_class_idx=3, m_steps=120)

    img = plot_img_attributions_and_save(image=img_name_tensors,
                                         baseline=baseline,
                                         target_class_idx=3,
                                         m_steps=400,
                                         cmap=plt.cm.inferno,
                                         overlay_alpha=0.4)

    return pred_label1, img


def compute_gradients(images, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = DiseaseDetectionModel(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, images)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_image', methods=['POST'])
def process_image_api():
    # Get the image from the request
    image_file = request.files['image']

    # Save the image to a temporary file
    temp_image_path = 'temp_image.jpg'
    image_file.save(temp_image_path)

    # Process the image
    label, processed_image = process_image(temp_image_path)

    buffered = BytesIO()
    processed_image.save(buffered, format="PNG")
    processed_image_bytes = buffered.getvalue()

    # Encode the image bytes as base64
    encoded_image = base64.b64encode(processed_image_bytes).decode('utf-8')

    # Return a JSON response with both the encoded image and the label
    response_data = {
        'encoded_image': encoded_image,
        'label': label
    }

    return jsonify(response_data)


if __name__ == '__main__':
    app.run(debug=True)
