from keras.applications import MobileNet, EfficientNetB0, efficientnet, MobileNetV3Small
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import cv2 as cv
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from keras.models import load_model
from skimage.segmentation import mark_boundaries
from keras.preprocessing.image import image_utils

num_classes = 3
folder_path = 'D:/Research/PlantVillage/Potato/Short/'
model_name = 'potato_model.h5'
image_name = '39.jpg'
image_size = (224, 224)


def readImage(imagePath):
    input_image = image_utils.load_img(imagePath, target_size=image_size)
    # Convert the image to a numpy array
    x = image_utils.img_to_array(input_image)
    # Expand the dimensions of the array (to include a batch dimension)
    # x = np.expand_dims(x, axis=0)
    # Preprocess the image using the MobileNet preprocessing function
    x = efficientnet.preprocess_input(x)
    return x


def getImageData():
    imageData = []
    image_count = num_classes * 100
    for i in range(image_count):
        img_path = folder_path + str(i) + '.jpg'
        img = readImage(img_path)
        imageData.append(img)

    imageData = np.array(imageData)
    print(imageData.shape)
    cls = [int(i / 100) for i in range(image_count)]
    cls = np.array(cls, dtype='int')

    train_data, test_data, train_labels, test_labels = train_test_split(imageData, cls, test_size=0.2, random_state=10)

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return train_data, test_data, train_labels, test_labels


def getModel():
    model = EfficientNetB0(include_top=True, weights='D:/Research/Weights_for_Transfer_Learning/efficientnetb0.h5')
    # model = MobileNetV3Small(include_top=True, weights='D:/Research/Weights_for_Transfer_Learning/weights_mobilenet_v3_small_224_1.0_float.h5')
    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Dense(1024, activation='relu', name='dd1')(x)
    x = Dropout(0.2, name='d1')(x)
    x = Dense(512, activation='relu', name='dd2')(x)
    x = Dropout(0.2, name='d2')(x)
    x = Dense(64, activation='relu', name='dd3')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=predictions)

    return model


def trainAndSaveModel():
    train_data, test_data, train_labels, test_labels = getImageData()
    model = getModel()

    optimizer = RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # Train the model
    model.fit(train_data, train_labels, epochs=90, batch_size=20, validation_data=(test_data, test_labels))
    model.save(model_name)


def explainDecision():
    model = load_model(model_name)
    # Load the input image
    image = readImage(folder_path + image_name)

    # Create a LIME explainer
    explainer = lime_image.LimeImageExplainer()

    # Generate an explanation for the image
    explanation = explainer.explain_instance(image, model.predict, top_labels=3, hide_color=0, num_samples=5000)
    print(explanation.top_labels[0:10])

    input_image = image_utils.load_img(folder_path + image_name, target_size=image_size)
    # Get the LIME mask for the top predicted class
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=25,
                                                hide_rest=True)

    # Overlay the LIME mask on the original image
    # masked_image = mark_boundaries(temp / 2 + 0.5, mask)
    original_img = np.array(input_image)
    masked_image = mark_boundaries(original_img / 255, mask, color=(1, 0, 0))

    # Display the original image and the LIME explanation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(input_image)
    ax1.set_title('Original Image')
    ax2.imshow(masked_image)
    ax2.set_title('LIME Explanation')
    plt.show()


'''def explainDecision_topN(N):
    model = load_model(model_name)
    # Load the input image
    image = readImage(folder_path + image_name)

    # Create a LIME explainer
    explainer = lime_image.LimeImageExplainer()

    # Generate an explanation for the image
    explanation = explainer.explain_instance(image, model.predict, top_labels=5, hide_color=0, num_samples=1000)
    top_labels = explanation.top_labels[0:N]
    print(explanation.score)
    #print(dir(explanation))

    input_image = image_utils.load_img(folder_path + image_name, target_size=image_size)
    fig, axis = plt.subplots(1, N+1, figsize=(10, 5))
    # Get the LIME mask for the top predicted class
    axis[0].imshow(input_image)
    axis[0].set_title('Original Image')

    for i in range(N):
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[i], positive_only=True, num_features=1, hide_rest=True)
        # Overlay the LIME mask on the original image
        masked_image = mark_boundaries(temp / 2 + 0.5, mask)
        # Display the original image and the LIME explanation
        axis[i+1].imshow(masked_image)
        axis[i+1].set_title('LIME Explanation for Class : ' + str(top_labels[i]))
    plt.show()
'''

if __name__ == '__main__':
    # getModel()
    # trainAndSaveModel()
    explainDecision()
    # explainDecision_topN(3)
    # img = readImage(folder_path + image_name)
    # print(np.min(img), np.max(img))
