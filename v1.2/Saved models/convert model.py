import tensorflow as tf
import keras
from keras import backend as K


# Load the Keras model saved in HDF5 format

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
loaded_model = keras.models.load_model(
    "Y:\Coding\Project\Apple\Plant Disease Detection\\v1.2\Saved models\SavedModelPlantDetectionMLModel3.h5",
    custom_objects=custom_objects)

# Convert and save the model in protobuf format
tf.saved_model.save(loaded_model, 'Y:\Coding\Project\Apple\Plant Disease Detection\\v1.2\Saved models')
