# %%
import time
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Sequential
import pickle
import numpy as np
from numba import jit, njit
import tensorflow as tf
import os
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# %%

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(
    label_mode="fine")
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
%matplotlib inline

imgplot = plt.imshow(x_train[18])
plt.show()


# %%

print(x_train.shape)
model = Sequential([])
# convolutional layer
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
          padding='valid', activation='relu', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Dropout(0.6))
model.add(MaxPool2D(pool_size=(1)))
model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
          padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.6))
model.add(MaxPool2D(pool_size=(1)))

model.add(Conv2D(100, kernel_size=(3, 3), strides=(1, 1),
          padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.6))
model.add(MaxPool2D(pool_size=(1)))
# flatten output of conv
model.add(Flatten())
model.add(Dropout(0.8))
# hidden layer
#model.add(Dense(300, activation='relu'))
# model.add(Dropout(0.7))
# output layer
model.add(Dense(100, activation='softmax'))
# use adam optimizer and categorical cross entropy cost

model.compile(optimizer="adam",
              loss="categorical_crossentropy", metrics=["acc"])
# after each epoch decrease learning rate by 0.95
#annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

# train
epochs = 300
j = 0
start_time = time.time()
with tf.device('/GPU:0'):
    history = model.fit(x_train, y_train, epochs=epochs,
                        validation_data=(x_test, y_test), batch_size=800)
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
# %%

tf.debugging.set_log_device_placement(True)


# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)

# %%
