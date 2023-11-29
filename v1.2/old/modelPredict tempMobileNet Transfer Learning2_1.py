from keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import load_model

# Load the saved model
loaded_model = load_model("Y:\Coding\Pycharm ML DL\SIH\\v1.1\\tempMobileNet Transfer Learning2_1.h5")

from keras.preprocessing import image
import numpy as np

# Load and preprocess the image to be predicted
img_path = 'Y:\Coding\Pycharm ML DL\SIH\\v1.1\dataset\\test\\test\AppleCedarRust1.JPG'  # Replace with the path to your image
img = image.load_img(img_path, target_size=(256, 256))
img = image.img_to_array(img)
print(img.shape)
img = np.expand_dims(img, axis=0)
print(img.shape)
img = preprocess_input(img)
print(img.shape)

# Make predictions
predictions = loaded_model.predict(img)

# Get the predicted class
predicted_class_index = np.argmax(predictions)

class_labels = ['apple scab', 'apple rot', 'apple cedar rust', 'apple healthy']  # Replace with your actual class labels
predicted_class = class_labels[predicted_class_index]
print(f"The predicted class is: {predicted_class}")
