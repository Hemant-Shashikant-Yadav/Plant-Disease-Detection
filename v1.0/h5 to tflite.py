import tensorflow as tf

# Load your Keras model
model = tf.keras.models.load_model("tempMobileNet Transfer Learning2_1.h5")

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open("../../../Program-backup/Pycharm ML DL/SIH/v1.1/model.tflite", "wb") as f:
    f.write(tflite_model)
