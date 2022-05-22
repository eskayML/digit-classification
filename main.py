
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import img_to_array,load_img
model = keras.models.load_model('./first_convnet_model_with_mnist_digit.h5')

print(model.summary())
image_path = 'test_images/img_0.PNG'

im = cv2.imread(image_path)
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im = cv2.resize(im, (28, 28))
# cv2.imshow('Number', im)


######### NORMALIZE DATA
im = cv2.bitwise_not(im)
im = im/255
print('Data Type of image', type(im))
im = im.reshape(-1, 28, 28, 1)
prediction = model.predict(im)
print(prediction)
print()
print("The number is:", np.argmax(prediction))

