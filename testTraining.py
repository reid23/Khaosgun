# %%
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


X_train = np.load(r'/home/reid/Documents/khaosTrainData/X_train1.npy')
X_test = np.load(r'/home/reid/Documents/khaosTrainData/X_test1.npy')
y_train = np.load(r'/home/reid/Documents/khaosTrainData/y_train1.npy')
y_test = np.load(r'/home/reid/Documents/khaosTrainData/y_test1.npy')

print("data loaded")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


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


print(X_train.shape)
X_train = addDimension(X_train)
X_test = addDimension(X_test)
print(X_train.shape)

# create multiple cnn model for ensembling
# model 1
# %%
# making the model

model = Sequential([])
# convolutional layer
model.add(Conv2D(100, kernel_size=(3, 3), strides=(1, 1),
          padding='valid', activation='relu', input_shape=(128, 128, 1)))
model.add(BatchNormalization())
model.add(Dropout(0.6))
model.add(Conv2D(100, kernel_size=(7, 7), strides=(1, 1),
          padding='valid', activation='relu'))
model.add(Dropout(0.6))
model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
          padding='valid', activation='relu'))
model.add(Dropout(0.6))
#model.add(MaxPool2D(pool_size=5))

model.add(Conv2D(32, kernel_size=(5, 5), strides=(2, 2),
          padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.6))
#model.add(MaxPool2D(pool_size=5))
model.add(Dense(128))
model.add(Dropout(0.4))
model.add(Dense(64))
model.add(Dropout(0.4))
model.add(Dense(32))
model.add(Dropout(0.5))
# flatten output of conv
model.add(Flatten())
model.add(Dropout(0.3))
# hidden layer
#model.add(Dense(300, activation='relu'))
# model.add(Dropout(0.7))
# output layer
model.add(Dense(2, activation='softmax'))
# use adam optimizer and categorical cross entropy cost

model.compile(optimizer="adam",
              loss="categorical_crossentropy", metrics=["acc"])
# after each epoch decrease learning rate by 0.95
#annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

# %%
# train

plt.imshow(X_train[0], cmap='gray')
plt.show()
epochs = 1000
j = 0
start_time = time.time()
with tf.device('/GPU:0'):
    history = model.fit(X_train, y_train, epochs=epochs,
                        validation_data=(X_test, y_test), batch_size=10)
end_time = time.time()
# print_time_taken(start_time, end_time)


print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
    j+1, epochs, history.history['acc'][epochs-1], history.history['val_acc'][epochs-1]))

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('withPreprocessing.h5')
# %%
