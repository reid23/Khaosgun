#%%
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
import os
import random
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.datasets import mnist
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from PIL import Image
import time
import h5py
import matplotlib.pyplot as plt
from numba import njit
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

@njit
def addDimension(arrayOfImages):
    iCounter = 0
    jCounter = 0
    kCounter = 0
    output = np.empty(
        (arrayOfImages.shape[0], arrayOfImages.shape[1], arrayOfImages.shape[2], 1))
    for i in arrayOfImages:
        for j in i:
            for k in j:
                output[iCounter][jCounter][kCounter] = k
                kCounter += 1
            kCounter = 0
            jCounter += 1
        jCounter = 0
        iCounter += 1
    iCounter = 0
    return output
    print("done!")

model = load_model('withPreprocessing2.h5')

X_train = np.load(r'/home/reid/Documents/khaosTrainData/X_train.npy')
X_test = np.load(r'/home/reid/Documents/khaosTrainData/X_test.npy')
y_train = np.load(r'/home/reid/Documents/khaosTrainData/y_train.npy')
y_test = np.load(r'/home/reid/Documents/khaosTrainData/y_test.npy')

print(X_train.shape)
X_train = addDimension(X_train)
X_test = addDimension(X_test)
print(X_train.shape)

#%%
counter= 0
startTime = time.time()
for j in range(100):
    for i in model.predict(X_test):
        '''
        plt.imshow(X_test[counter], cmap='gray')
        plt.annotate(str(i), xy=(0,10), color='green')
        plt.show()
        '''
        counter += 1

endTime = time.time()

print(f'time taken for {str(counter)} inferences: {str(endTime-startTime)}')
print(f'time per iteration: {str((endTime-startTime)/counter)}')
# %%
