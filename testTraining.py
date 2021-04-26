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

def getNumpyArray(dir,imageShape=(256,256,3)):
    img = Image.open(dir)
    img = img.resize((imageShape[0], imageShape[1]), Image.ANTIALIAS)
    output = np.array(img, dtype=np.float32)
    return np.array([output], dtype=np.float32)

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

model.add(Conv2D(32, kernel_size = 3, activation=tf.keras.layers.LeakyReLU(alpha=0.3), input_shape = (256, 256, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 3, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size = 3, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 3, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size = 5, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size = 5, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size = 7, strides=2, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
model.add(BatchNormalization())
model.add(Dropout(0.4))


model.add(Conv2D(256, kernel_size = 4, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(4, activation='softmax'))

# use adam optimizer and categorical cross entropy cost
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"]) 
# after each epoch decrease learning rate by 0.95
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.96 ** x)

# train
epochs = 100
j=0

checkpoint_filepath = 'checkpoints'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

start_time = time.time()
history = model.fit(X_train, y_train, epochs = epochs, validation_data=(X_test,y_test), batch_size=20, callbacks=[model_checkpoint_callback])
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


def run_model(image_dir, plot_image=True, cpu_only=False):
    startTime = time.time()
    if cpu_only==True:
        with tf.device("/cpu:0"):
            output_data = np.array(model.predict(getNumpyArray(image_dir)))
            endTime = time.time()
    else:
        output_data = np.array(model.predict(getNumpyArray(image_dir)))
        endTime = time.time()
    classification_time = endTime-startTime
    print(image_dir + ":")
    if plot_image==True:
        imgplot = plt.imshow(getNumpyArray(image_dir)[0]/255)
        plt.show()
    print("person: " + str(np.format_float_positional(output_data[0][0], trim = '-')))
    print("nothing: " + str(np.format_float_positional(output_data[0][1], trim = '-')))
    print("squirrel: " + str(np.format_float_positional(output_data[0][2], trim = '-')))
    print("chicken: " + str(np.format_float_positional(output_data[0][3], trim = '-')))
    print("classification time: " + str(classification_time) + " seconds")
    print("---------------------------------------------------------------------")
run_model("buisness.jpeg", plot_image = False)
run_model("images.jpg", plot_image = False)