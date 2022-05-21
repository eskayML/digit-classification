
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
model = keras.models.load_model('./digit_classification.h5')

print(model.summary())
image_path = 'test_images/img_3.PNG'
im = cv2.imread(image_path)
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im = cv2.resize(im, (28, 28))
# cv2.imshow('Number', im)




######### NORMALIZE DATA
im = cv2.bitwise_not(im)
print('Data Type of image', type(im))
im = im.reshape(-1, 28, 28, 1)
prediction = model.predict(im)
print(prediction)
print()
print("The number is:", np.argmax(prediction))

