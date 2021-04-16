import tensorflow as tf
import numpy as np

def representative_dataset():
    for _ in range(255):
      data = np.random.rand(1, 128, 128, 3)
      yield [data.astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_saved_model('C:\\Users\\reidd\\Khaosgun')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
	f.write(tflite_model)