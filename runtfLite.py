import numpy as np
from tflite_runtime.interpreter import Interpreter
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

def getNumpyArray(dir,imageShape=(128,128,3)):
    img = Image.open(dir)
    img = img.resize((imageShape[0], imageShape[1]), Image.ANTIALIAS)
    output = np.array(img, dtype=np.float32)
    return np.array([output], dtype=np.float32)
# Load the TFLite model and allocate tensors.
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# get input data
input_data = getNumpyArray("man.jpg")

#classify
startTime = time.time()
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
endTime = time.time()

classification_time = np.round(endTime-startTime, 3)



class TestModel(tf.Module):
  def __init__(self):
    super(TestModel, self).__init__()

  @tf.function(input_signature=[tf.TensorSpec(shape=[1, 10], dtype=tf.float32)])
  def add(self, x):
    '''
    Simple method that accepts single input 'x' and returns 'x' + 4.
    '''
    # Name the output 'result' for convenience.
    return {'result' : x + 4}

# Save the model
module = TestModel()
# You can omit the signatures argument and a default signature name will be
# created with name 'serving_default'.
tf.saved_model.save(
    module, SAVED_MODEL_PATH,
    signatures={'my_signature':module.add.get_concrete_function()})

# Convert the model using TFLiteConverter
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
tflite_model = converter.convert()
with open(TFLITE_FILE_PATH, 'wb') as f:
  f.write(tflite_model)

# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)
# There is only 1 signature defined in the model,
# so it will return it by default.
# If there are multiple signatures then we can pass the name.
my_signature = interpreter.get_signature_runner()

# my_signature is callable with input as arguments.
output = my_signature(x=tf.constant([1.0], shape=(1,10), dtype=tf.float32))
# 'output' is dictionary with all outputs from the inference.
# In this case we have single output 'result'.
print(output['result'])



print("person: " + str(np.format_float_positional(output_data[0][0], trim = '-')))
print("nothing: " + str(np.format_float_positional(output_data[0][1], trim = '-')))
print("squirrel: " + str(np.format_float_positional(output_data[0][2], trim = '-')))
print("cat: " + str(np.format_float_positional(output_data[0][3], trim = '-')))
print("chicken: " + str(np.format_float_positional(output_data[0][4], trim = '-')))
print("classification time: " + str(classification_time) + " seconds")
img = input_data[0]
imgplot = plt.imshow(img/255)
plt.show()