from keras.applications.mobilenet import preprocess_input
from keras.models import load_model
from keras import backend as K
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries


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
img_path = 'Y:\Coding\Project\Apple\Plant Disease Detection\Datasets\Full dataset\\test\\test\AppleCedarRust2.JPG'  # Replace with the path to your image
# img = image.load_img(img_path, target_size=(256, 256))
img = image.load_img(img_path)

img1 = image.img_to_array(img)
img = image.img_to_array(img)

# print(img.shape)
img = np.expand_dims(img, axis=0)
# print(img.shape)
img = preprocess_input(img)
img1 = preprocess_input(img1)
# print(img.shape)

# Make predictions
predictions = loaded_model.predict(img)

# Get the predicted class
predicted_class_index = np.argmax(predictions)

class_labels = ['apple scab', 'apple rot', 'apple cedar rust', 'apple healthy']  # Replace with your actual class labels
predicted_class = class_labels[predicted_class_index]
print(f"The predicted class is: {predicted_class}")

from lime.lime_image import LimeImageExplainer
from keras.preprocessing.image import load_img


def explainDecision():
    # Load the input image

    # Create a LIME explainer
    explainer = LimeImageExplainer()

    # Generate an explanation for the image
    explanation = explainer.explain_instance(img1, loaded_model.predict, top_labels=4, hide_color=0, num_samples=5000)
    print(explanation.top_labels[0:10])

    input_image = load_img(img_path)
    # Get the LIME mask for the top predicted class
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=25,
                                                hide_rest=True)

    # Overlay the LIME mask on the original image
    masked_image = mark_boundaries(temp / 2 + 0.5, mask)
    original_img = np.array(input_image)
    masked_image = mark_boundaries(original_img / 256, mask, color=(1, 0, 0))

    # Display the original image and the LIME explanation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(input_image)
    ax1.set_title('Original Image')
    ax2.imshow(masked_image)
    ax2.set_title('LIME Explanation')
    plt.show()


explainDecision()
