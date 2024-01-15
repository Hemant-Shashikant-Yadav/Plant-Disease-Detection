import tensorflow as tf

# Load your Keras model
model = tf.keras.models.load_model("tempMobileNet Transfer Learning2_1.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("../../../Program-backup/Pycharm ML DL/SIH/v1.1/model.tflite", "wb") as f:
    f.write(tflite_model)
