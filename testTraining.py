import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten ,Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
import os
import random
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
from PIL import Image
import time
import h5py
import matplotlib.pyplot as plt
import keras
from keras import optimizers


X_train = np.load(r'D:\Documents\X_train.npy')
X_test = np.load(r'D:\Documents\X_test.npy')
y_train = np.load(r'D:\Documents\y_train.npy')
y_test = np.load(r'D:\Documents\y_test.npy')

print("data loaded")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model = Sequential()


model.add(Conv2D(128, kernel_size = 3, activation='relu', input_shape = (128, 128, 3)))
model.add(Conv2D(128, kernel_size = 7, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.6))

model.add(Conv2D(32, kernel_size = 3, activation='relu'))
model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(8, kernel_size = 4, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"]) 

# after each epoch decrease learning rate by 0.95
#annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

# train
epochs = 200
j=0
start_time = time.time()
history = model.fit(X_train, y_train, epochs = epochs, validation_data=(X_test,y_test), batch_size=50)
end_time = time.time()
#print_time_taken(start_time, end_time)



print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(j+1,epochs,history.history['acc'][epochs-1],history.history['val_acc'][epochs-1]))

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("trainingHistory.png")

model.save('C:\\Users\\reidd\\Khaosgun')