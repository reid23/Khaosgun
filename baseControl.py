#oall control

import RPi.GPIO as GPIO
import time
import os
import subprocess
from picamera import PiCamera
from enum import Enum
from PIL import Image
import keras
#import tensorflow as tf
import numpy as np
#from PIL import Image
import cv2
import h5py
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
import io
from time import sleep
import picamera
import os
import keras


GPIO.setmode(GPIO.BCM)

GPIO.setup(6, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(16, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(22, GPIO.IN, pull_up_down=GPIO.PUD_UP)

photoNumber = 0
videoNumber = 0
iscamera = False

#variables and comments
#aividMode     #False means squirrel/photo, True means human/video
#pbState       #the state of the photo button.  False means pressed, True means unpressed
#oldpbState    #the state of the photo button last iteration, to make sure action is only take$
#controlState  #0 means none, 1 means web, 2 means AI
#oldControlState

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

recording = False

videoNumber = 0
photoNumber = 0
def loadImage(path):
    img = cv2.imread(path)
    #print('Original Dimensions : ',img.shape)
    
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
def takeWebPhoto():
    os.system("/var/www/html/photo.sh &")
def takeAIPhoto():
    if(aividMode == True):
        takeWebPhoto();
    else:
        pass # only works with human detection currently
        
    #AI photo is the same as web photo
def takePhoto():
    global photoNumber
    global camera
    camera.capture("/var/www/html/photos/Photo{}.jpg".format(photoNumber))
    photoNumber+=1
def videoAction():
    global videoNumber
    global camera
    global recording
    if(recording == False):
        camera.start_recording("/var/www/html/photos/Video{}.h264".format(videoNumber))
        videoNumber+=1
        recording = True
        print("recording...")
    elif(recording == True):
        camera.stop_recording()
        recording = False
        print("stopping recording")
def startAI():
    pastTen = np.array([0,0,0,0,0,0,0,0,0,0])
    global camera
    print("starting AI control")
#    os.system("sudo kill -9 `pidof mjpg_streamer` &")
    global iscamera
    if(aividMode == True):
        os.system("sudo kill -9 `pidof mjpg_streamer` &")
        os.system('rm -rf /var/www/html/squirrel.html')
        os.system('touch /var/www/html/human.html')
        os.system("sudo kill -9 `pidof mjpg_streamer` &")
        camera.stop_preveiw()
        camera.close()
        iscamera = False
    else:
        os.system('rm -rf /var/www/html/human.html')
        os.system('touch /var/www/html/squirrel.html')
        time.sleep(0.5)
        camera = PiCamera()
        camera.start_preveiw()
        iscamera = True
def startWebControl():
    global camera
    print("starting web control")
    global iscamera
    if(iscamera == True):
        camera.stop_preveiw()
        camera.close()
        time.sleep(0.2)
        iscamera == False
    os.system("sudo /usr/local/bin/mjpg_streamer -i 'input_uvc.so -r 1280x720 -d /dev/video0 -f 30 -q 80' -o 'output_http.so -p 8080 -w /usr/local/share/mjpg-streamer/www' &")
    os.system('rm -rf /var/www/html/squirrel.html; rm -rf /var/www/html/human.html')
def startManual():
    global camera
    os.system('rm -rf /var/www/html/squirrel.html; rm -rf /var/www/html/human.html')
    print("starting manual mode")
    global iscamera
    os.system("sudo kill -9 `pidof mjpg_streamer` &")
    time.sleep(0.5)
    camera = PiCamera()
    camera.start_preveiw()
    iscamera = True

pbState = True
controlState = 4
aividMode = False
detectionThreshold = 0.5
while True:
    oldaividMode = aividMode
    aividMode = GPIO.input(16)
    oldpbState = pbState
    pbState = GPIO.input(6)
    oldControlState = controlState
    if(GPIO.input(27) == False):
        controlState = 1
    elif(GPIO.input(22) == False):
        controlState = 2
    else:
        controlState = 0

    if(pbState == oldpbState):
        pass
    else:
        if(pbState == True):
            if(aividMode == False):
                if(controlState == 0):
                    takePhoto()
                elif(controlState == 1):
                    takeWebPhoto()
                else:
                    takeAIPhoto()
            else:
                if(controlState == 0):
                    videoAction()
                elif(controlState == 2):
                    pass
    if(oldControlState == controlState):
        pass
    else:
        if(controlState == 0):
            startManual()
        elif(controlState == 1):
            startWebControl()
        elif(controlState == 2):
            startAI()
        else:
            startManual()
    if(aividMode == oldaividMode):
        pass
    else:
        if(aividMode == True):
            os.system("sudo kill -9 `pidof mjpg_streamer` &")
            os.system('rm -rf /var/www/html/squirrel.html')
            os.system('touch /var/www/html/human.html')
            os.system("sudo kill -9 `pidof mjpg_streamer` &")
            camera.stop_preveiw()
            camera.close()
            iscamera = False
        else:
            os.system('rm -rf /var/www/html/human.html')
            os.system('touch /var/www/html/squirrel.html')
            time.sleep(0.5)
            camera = PiCamera()
            camera.start_preveiw()
            iscamera = True
    if(controlState == 2):

        camera.capture('img.jpg')
        img = loadImage('img.jpg')
        pred = detectClasses(img)
        np.append(pastTen,[pred[0][0]])
        np.delete(pastTen,[0])
        if(np.mean(pastTen) >= detectionThreshold):
            shoot()
        else:
            stopShoot()