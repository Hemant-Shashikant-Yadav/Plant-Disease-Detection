import base64
import io
from flask import Flask, request, render_template
from keras.applications.mobilenet import preprocess_input
from keras.models import load_model
from keras import backend as K
from matplotlib import pyplot as pltp

plt.switch_backend('agg')
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from flask import jsonify
from PIL import Image
import tensorflow as tf
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


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


def load_models():
    custom_objects = {'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m}

    global PlantValidationModel, DiseaseDetectionLables, DiseaseDetectionModel

    PlantValidationModel = load_model(
        "Y:\Coding\Project\Apple\Plant Disease Detection\\v1.2\Python files\Plant Validation Model\PlantValiidationSavedModel.h5",
        custom_objects=custom_objects)

    DiseaseDetectionLables = ['Apple Scab', 'Apple Rot', 'Apple Cedar Rust', 'Apple Healthy']

    DiseaseDetectionModel = load_model(
        "Y:\Coding\Project\Apple\Plant Disease Detection\\v1.2\Saved models\SavedModelPlantDetectionMLModel3.h5",
        custom_objects=custom_objects)


def preprocess_image(image1):
    if image1.mode != "RGB":
        image1 = image1.convert("RGB")
    image1 = image1.resize((256, 256))
    image1 = img_to_array(image1)
    image1 = np.expand_dims(image1, axis=0)
    image1 = preprocess_input(image1)
    return image1


def interpolate_images(baseline, image, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x + alphas_x * delta
    return images


def compute_gradients(images, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = DiseaseDetectionModel(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, images)


def integral_approximation(gradients):
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


def plot_img_attributions_and_return(baseline, image, target_class_idx, m_steps=20, cmap=None, overlay_alpha=0.4):
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

    # for i, ax in enumerate(axs.flat):
    #     buffer = io.BytesIO()
    #     plt.savefig(buffer, format='png')
    #     buffer.seek(0)
    #     image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    #     images_base64[f'image_{i+1}'] = image_base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    images_base64 = image_base64

    plt.close()  # Close the figure to avoid displaying the plot

    return images_base64




@app.route("/predict", methods=["POST"])
def predict():
    image_file = request.files['image']
    base64_image = request.form['base64Image']
    image = Image.open(image_file)

    load_models()

    processed_image = preprocess_image(image)

    prediction = PlantValidationModel.predict(processed_image).tolist()
    predicted_class_index = np.argmax(prediction)
    if (predicted_class_index == 1):
        print("You inputed wrong image. This is not a plant.")
        response = jsonify({
            'prediction': {
                'plantLabel': "You inputed wrong image. This is not a plant.",
                'imagefile': base64_image}
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    prediction1 = DiseaseDetectionModel.predict(processed_image).tolist()
    predicted_class_index1 = np.argmax(prediction1)
    plant_lable = DiseaseDetectionLables[predicted_class_index1]
    print(plant_lable)
    for i in range(len(DiseaseDetectionLables)):
        confidence = prediction1[0][i]
        print(f"Confidence for {DiseaseDetectionLables[i]}: {confidence * 100:.2f}%")

    image_array = np.array(image)

    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    eager_tensor = tf.convert_to_tensor(image_array, dtype=tf.uint8)

    img_name_tensors = read_image(eager_tensor)

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

    xai = plot_img_attributions_and_return(image=img_name_tensors,
                                           baseline=baseline,
                                           target_class_idx=3,
                                           m_steps=400,
                                           cmap=plt.cm.inferno,
                                           overlay_alpha=0.4)

    response = jsonify({
        'prediction': {
            'plantLabel': plant_lable,
            'imagefile': xai}
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    app.run(host='192.168.29.177', port=4000)
