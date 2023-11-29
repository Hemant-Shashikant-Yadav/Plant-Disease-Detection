import keras
from keras import backend as K
from keras.layers.core import Dense, Activation, Dropout

from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input, MobileNet
import numpy as np
from IPython.display import Image
# import matplotlib_inline as plt

from tensorflow.keras.optimizers import Adam

mobile = keras.applications.mobilenet.MobileNet()

base_model = MobileNet(weights='imagenet',
                       include_top=False)  # imports the mobilenet model and discards the last 1000 neuron layer.

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(
    x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)  # dense layer 2
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)  # dense layer 3
preds = Dense(4, activation='softmax')(x)  # final layer with softmax activation

model = Model(inputs=base_model.input, outputs=preds)

for i, layer in enumerate(model.layers):
    print(i, layer.name)

for layer in model.layers:
    layer.trainable = False
# or if we want to set the first 20 layers of the network to be non-trainable
for layer in model.layers[:20]:
    layer.trainable = False
for layer in model.layers[20:]:
    layer.trainable = True

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # included in our dependencies

train_generator = train_datagen.flow_from_directory(
    'Y:\Coding\Pycharm ML DL\SIH\\v1.1\dataset\PlantDiseasesDataset\\train', target_size=(256, 256), color_mode='rgb',
    batch_size=32, class_mode='categorical', shuffle=True)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory('Y:\Coding\Pycharm ML DL\SIH\\v1.1\dataset\\test',
                                                  target_size=(100, 100), color_mode='rgb', shuffle=False)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

step_size_train = train_generator.n // train_generator.batch_size
model.fit_generator(generator=train_generator, steps_per_epoch=step_size_train, epochs=10)

results = model.evaluate(test_generator)

print(results)


def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(100, 100))
    img_tensor = image.img_to_array(img)  # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor,
                                axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.  # imshow expects values in the range [0, 1]
    return img_tensor


img_path = ('Y:\Coding\Pycharm ML DL\SIH\\v1.1\dataset\\test\\test\AppleCedarRust4.JPG')
new_image = load_image(img_path)

pred = model.predict(new_image)
print(pred)

img_path = (
    'Y:\Coding\Pycharm ML DL\SIH\\v1.1\dataset\PlantDiseasesDataset\\train\Apple___Apple_scab\\0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417_90deg.JPG')
new_image = load_image(img_path)

pred = model.predict(new_image)
print(pred)

img_path = (
    'Y:\Coding\Pycharm ML DL\SIH\\v1.1\dataset\PlantDiseasesDataset\\valid\Apple___Cedar_apple_rust\\0a41c25a-f9a6-4c34-8e5c-7f89a6ac4c40___FREC_C.Rust 9807_new30degFlipTB.JPG')
new_image = load_image(img_path)

pred = model.predict(new_image)
print(pred)
