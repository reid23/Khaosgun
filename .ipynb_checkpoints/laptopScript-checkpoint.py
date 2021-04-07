import os
import http.client
import time
from PIL import Image
import tempfile
import subprocess
import urllib.request
import requests
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
import keras

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

model.load_weights(r'model_3.h5')

def req(url):
    r = requests.get(url)
    return r.text

def urlIsAlive(url):
    request = urllib.request.Request(url)
    request.get_method = lambda: 'HEAD'
    try:
        urllib.request.urlopen(request)
        return True
    except urllib.request.HTTPError:
        return False

def fire():
    print("shoot!!!!!!")
    req("http://khaosgun.local/pinon.php")
    time.sleep(0.1)  # this is how long the gun revs to get going before firing.
    req("http://khaosgun.local/firepinon.php")

def stopFire():
    req("http://khaosgun.local/firepinoff.php")
    req("http://khaosgun.local/pinoff.php")

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

#pastTen = np.array([0,0,0,0,0,0,0,0,0,0])
detectionThreshold = 0.6
while True:
    os.system("wget http://khaosgun.local:8080/?action=snapshot -O /Users/reiddye/darknet/data/mjpgStream.jpg")
    with tempfile.TemporaryFile() as tempf:
        if(urlIsAlive("http://khaosgun.local/squirrel.html")):
            im = Image.open("/Users/reiddye/darknet/data/mjpgStream.jpg")
            width, height = im.size

            # Setting the points for cropped image

            # oall x: 1280
            # oall y: 720
            #2556 × 1440

            top = (850/2556) * width  # x1
            left = (1070/1440) * height  # y1

            right = (1415/2556) * width  # x2
            bottom = (1286/1440) * height  # y2

            # Cropped image of above dimension
            im = im.crop((left, top, right, bottom))

            # Shows the image in image viewer
            im.save("/Users/reiddye/darknet/data/mjpgStream.jpg")
            pred = detectClasses(loadImage("/Users/reiddye/darknet/data/mjpgStream.jpg"))
            if(pred[0][0] >= detectionThreshold):
                fire()
            else:
                stopFire()
        elif(urlIsAlive("http://khaosgun.local/human.html")):
            im = Image.open("/Users/reiddye/darknet/data/mjpgStream.jpg")
            width, height = im.size

             # Setting the points for cropped image
            top = (250/2556) * width  # x1
            left = (800/1440) * height  # y1

            right = (1665/2556) * width  # x2
            bottom = (1440/1440) * height  # y2

            # Cropped image of above dimension
            im = im.crop((left, top, right, bottom))

            # saves the cropped image
            im.save("/Users/reiddye/darknet/data/mjpgStream.jpg")


            proc = subprocess.Popen(['./detectHuman.sh'], stdout=tempf)
            proc.wait()
            tempf.seek(0)
            result = tempf.read()
            result = str(result)
            print(result)
            numberIndex = result.find('person')
            if(numberIndex != -1):
                numberIndex += 8
                result = result[numberIndex:numberIndex+2]
                if(result.find("%") != -1):
                    result = result[:1]
                if(float(result)>20): #change the 20 for changing sensetivity
                    fire()
                print(result)
            else:
                stopFire()
                
        else:
            stopFire()
