
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
model = keras.models.load_model('./digit_classification.h5')

image_path = 'test_images/img_1.png'
im = cv2.imread(image_path)
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im = cv2.resize(im, (28, 28))

# normalize data
im = im/255
im = np.array(im)
print('The array is ', end='')
print(im.shape)
print(im)
prediction = model.predict([im])
print(prediction)
cv2.imshow('Image', im)
cv2.waitKey(0)
cv2.destroyAllWindows()
