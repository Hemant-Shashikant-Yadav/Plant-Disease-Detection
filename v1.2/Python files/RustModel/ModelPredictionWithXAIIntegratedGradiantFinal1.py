from io import BytesIO

import tensorflow as tf
from matplotlib import pyplot as plt
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

img_path = 'Y:\Coding\Project\Apple\Plant Disease Detection\\v1.2\Python files\Prediction models\A.JPG'
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


########################################################################################################################

ig_attributions = integrated_gradients(baseline=baseline, image=img_name_tensors, target_class_idx=3, m_steps=120)


############################################################################################################################


def plot_img_attributions_and_save(baseline, image, target_class_idx, m_steps=20, cmap=None, overlay_alpha=0.4,
                                   save_path=None):
    attributions = integrated_gradients(baseline=baseline, image=image, target_class_idx=target_class_idx,
                                        m_steps=m_steps)

    attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

    fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))

    axs[0, 0].set_title('Baseline image')
    axs[0, 0].imshow(baseline)
    axs[0, 0].axis('off')

    axs[0, 1].set_title('Original image')
    axs[0, 1].imshow(image)
    axs[0, 1].axis('off')

    axs[1, 0].set_title('Attribution mask')
    axs[1, 0].imshow(attribution_mask, cmap=cmap or 'viridis')
    axs[1, 0].axis('off')

    axs[1, 1].set_title('Overlay')
    axs[1, 1].imshow(attribution_mask, cmap=cmap or 'viridis')
    axs[1, 1].imshow(image, alpha=overlay_alpha)
    axs[1, 1].axis('off')

    # Convert the images to base64-encoded strings
    images_base64 = {}
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    images_base64[f'image'] = image_base64

    # Save the JSON object to a local directory
    if save_path:
        json_path = save_path
        with open(json_path, 'w') as json_file:
            json.dump(images_base64, json_file, indent=4)

    plt.close()  # Close the figure to avoid displaying the plot

    return json_path


save_path = ".\\attributions_data.json"

############################################################################################################################################

_ = plot_img_attributions_and_save(image=img_name_tensors,
                                   baseline=baseline,
                                   target_class_idx=3,
                                   m_steps=400,
                                   cmap=plt.cm.inferno,
                                   overlay_alpha=0.4,
                                   save_path=save_path)
