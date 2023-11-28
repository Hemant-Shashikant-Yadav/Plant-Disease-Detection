import json
from keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K



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
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# Load the saved model
loaded_model = load_model("Y:\Coding\Jupyter\SIH\\v1.2\\tempMobileNet Transfer Learning111.h5",custom_objects={'f1_m': f1_m, 'precision_m':precision_m,'recall_m':recall_m})

# Create an ImageDataGenerator for the test Full Apple dataset
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load the test data using the generator
test_generator = test_datagen.flow_from_directory(
    'Y:\Coding\Jupyter\SIH\\v1.1\dataset\PlantDiseasesDataset\\valid',
    target_size=(256, 256),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model on the test Full Apple dataset
history_callback = loaded_model.evaluate(test_generator, verbose=1)

# Extract the evaluation metrics
loss, accuracy, f1_m, precision_m, recall_m = history_callback

# Print and save the evaluation metrics
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
print(f"F1 score: {f1_m}")
print(f"Precision: {precision_m}")
print(f"Recall: {recall_m}")

# Save the evaluation history to a JSON file
evaluation_history = {
    'loss': loss,
    'accuracy': accuracy,
    'f1_score': f1_m,
    'precision': precision_m,
    'recall': recall_m
}

with open('testing_history.json', 'w') as json_file:
    json.dump(evaluation_history, json_file)
