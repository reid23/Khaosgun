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
import h5py
from tensorflow import keras
model = keras.models.load_model(
    '/home/reid/Documents/repositories/Khaosgun/model_simple.h5')
model.summary()
