#import block
import keras
#import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import h5py
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,Activation,MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.datasets import mnist
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import matplotlib.pyplot as plt


#define model
model = Sequential()

model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (100, 100, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))


model.add(Conv2D(256, kernel_size = 4, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(3, activation='softmax'))
#end model


x_test = np.load(r'D:\Downloads\results\x_test.npy')
x_train = np.load(r'D:\Downloads\results\x_train.npy')
y_test = np.load(r'D:\Downloads\results\y_test.npy')
y_train = np.load(r'D:\Downloads\results\y_train.npy')
print(x_test[1].shape)
print("shape of labels:")
print(y_test)
#function to make stuff readable
labels = ['gatto', 'scoiattolo', 'humans']

#image loading/reshaping
img = cv2.imread('standing.jpg')
print('Original Dimensions : ',img.shape)
 
width = 100
height = 100
dim = (width, height)
 
#     resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)
 
cv2.imshow("Resized image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("resized image shape: ")
print(resized.shape)


#load from model folder
#model = keras.models.load_model('D:\Downloads\model (1)\content\model.model')

#load weights from saved h5
model.load_weights(r'D:\Downloads\results\model_3.h5')

print(resized.shape)
plt.imshow(resized)
plt.show()
pred = model.predict_classes(resized.reshape(-1,100,100,3))
print("prediction for local image")
print(pred)


im_list = [1,100,200,300,400,500,358,693]
for i in im_list:
#     i = 1000  #index from test data to be used, change this other value to see a different image
    img = x_test[i]
    print(img.shape)
    plt.imshow(img)
    plt.show()
    pred = model.predict_classes(img.reshape(-1,100,100,3))
    print("prediction")
    print(pred)
    
    actual =  y_test[i]
    
    
    print(f'actual: {actual}')
    print(f'predicted: {pred}')


#print(x_test.shape)