import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.models import load_model
from keras import backend as K

class_labels = ['apple scab', 'apple rot', 'apple cedar rust', 'apple healthy']


def read_image(file_name):
    image = tf.io.read_file(file_name)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_pad(image, target_height=256, target_width=256)
    return image


img_paths = {
    'A1': 'Y:\Coding\Project\Apple\Plant Disease Detection\Datasets\Apple dataset\\test\\test\AppleCedarRust2.JPG',
    'A2': 'Y:\Coding\Project\Apple\Plant Disease Detection\Datasets\Apple dataset\\test\\test\AppleCedarRust4.JPG',
}
img_name_tensors = {name: read_image(img_path) for (name, img_path) in img_paths.items()}


def top_k_predictions(img, k=4):
    image_batch = tf.expand_dims(img, 0)
    predictions = loaded_model(image_batch)
    probs = tf.nn.softmax(predictions, axis=-1)
    top_probs, top_idxs = tf.math.top_k(input=probs, k=k)
    return top_probs[0]


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
    "Y:\Coding\Project\Apple\Plant Disease Detection\\v1.2\Saved models\SavedModelPlantDetectionMLModel3.h5",
    custom_objects=custom_objects)

for (name, img_tensor) in img_name_tensors.items():
    pred_prob = top_k_predictions(img_tensor)
    print(pred_prob)

baseline = tf.zeros(shape=(256, 256, 3))

m_steps = 50
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


# fig = plt.figure(figsize=(20, 20))
#
# i = 0
# for alpha, image in zip(alphas[0::10], interpolated_images[0::10]):
#     i += 1
#     plt.subplot(1, len(alphas[0::10]), i)
#     plt.title(f'alpha: {alpha:.1f}')
#     plt.imshow(image)
#     plt.axis('off')
#
# plt.tight_layout();


def compute_gradients(images, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = loaded_model(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, images)


path_gradients = compute_gradients(
    images=interpolated_images,
    target_class_idx=4)


def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients


ig = integral_approximation(
    gradients=path_gradients)


def integrated_gradients(baseline, image, target_class_idx, m_steps=50, batch_size=32):
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


ig_attributions = integrated_gradients(baseline=baseline, image=img_name_tensors['A1'], target_class_idx=4, m_steps=240)

ig_attributions = integrated_gradients(baseline=baseline, image=img_name_tensors['A2'], target_class_idx=4, m_steps=240)


# @title
def plot_img_attributions(baseline, image, target_class_idx, m_steps=50, cmap=None, overlay_alpha=0.4):
    attributions = integrated_gradients(baseline=baseline, image=image, target_class_idx=target_class_idx,
                                        m_steps=m_steps)

    # Sum of the attributions across color channels for visualization.
    # The attribution mask shape is a grayscale image with height and width
    # equal to the original image.
    attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

    fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))

    axs[0, 0].set_title('Baseline image')
    axs[0, 0].imshow(baseline)
    axs[0, 0].axis('off')

    axs[0, 1].set_title('Original image')
    axs[0, 1].imshow(image)
    axs[0, 1].axis('off')

    axs[1, 0].set_title('Attribution mask')
    axs[1, 0].imshow(attribution_mask, cmap=cmap)
    axs[1, 0].axis('off')

    axs[1, 1].set_title('Overlay')
    axs[1, 1].imshow(attribution_mask, cmap=cmap)
    axs[1, 1].imshow(image, alpha=overlay_alpha)
    axs[1, 1].axis('off')

    plt.tight_layout()
    return fig
