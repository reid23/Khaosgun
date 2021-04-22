#this file is to dev the cnn to replace the YOLOv3 darknet model, at least for squirrels at the moment.
#yes, I should have used a branch
#but I just manually merged changes into baseControl when I was done
# so TLDR; it works

#import block
import keras
#import tensorflow as tf
import numpy as np
from PIL import Image
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
import picamera
import time
import os
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

model.load_weights(r'D:\Downloads\results\model_3.h5')

#image loading/reshaping
def loadImage(path):
    img = cv2.imread(path)
    print('Original Dimensions : ',img.shape)
    
    width = 100
    height = 100
    dim = (width, height)
    
    #     resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    resized = resized.reshape(-1,100,100,3)
    return resized

def detectClasses(img):
    pred = model.predict(img.reshape(-1,100,100,3))
    return pred
def shoot(revTimeSec = 0.2, holdTimeSec =  1.5):
    os.system("gpio -g mode 17 out")
    os.system("gpio -g write 17 0")
    time.sleep(revTimeSec)
    os.system("gpio -g mode 23 out")
    os.system("gpio -g write 23 0")
    time.sleep(holdTimeSec)

def stopShoot():
    os.system("gpio -g mode 17 out")
    os.system("gpio -g write 17 1")
    os.system("gpio -g mode 23 out")
    os.system("gpio -g write 23 1")



#pi stuff
detectionThreshold = 0.5

with picamera.PiCamera() as camera:
    camera.resolution = (1024, 768)
    camera.start_preview()
    # Camera warm-up time
    time.sleep(2)
    pastTen = np.array([0,0,0,0,0,0,0,0,0,0])
    
    while True:
        camera.capture('img.jpg')
        img = loadImage('img.jpg')
        pred = detectClasses(img)
        np.append(pastTen,[pred[0][0]])
        np.delete(pastTen,[0])
        if(np.mean(pastTen) >= detectionThreshold):
            shoot()
        else:
            stopShoot()

        

#load from model folder
#model = keras.models.load_model('D:\Downloads\model (1)\content\model.model')

#load weights from saved h5


#image = loadImage('images.jpg')
#plt.imshow(image.reshape(100,100,3))
#plt.show()
#pred = detectClasses(image)
#print("prediction for local image")
#print("squirrel      cat        human")
#print(pred)

#print(x_test.shape)