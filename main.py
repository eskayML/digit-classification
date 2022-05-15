
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
model = keras.models.load_model('./digit_classification.h5')

image_path = 'test_images/img_3.png'
im = cv2.imread(image_path)
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im = cv2.resize(im, (28, 28))


print("Model Summary")
print(model.summary())

# normalize data
im = im/255
im = np.array(im)
print('The array is ', end='')
print(im.shape)
print(im)
im = im.reshape(-1, 28, 28, 1)
prediction = model.predict(im)
print("The number is", np.argmax(prediction))
print(prediction)
