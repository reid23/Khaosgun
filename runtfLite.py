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
input_data = getNumpyArray("images.jpg")

#classify
startTime = time.time()
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
endTime = time.time()

classification_time = np.round(endTime-startTime, 3)



print("person: " + str(np.format_float_positional(output_data[0][0], trim = '-')))
print("nothing: " + str(np.format_float_positional(output_data[0][1], trim = '-')))
print("squirrel: " + str(np.format_float_positional(output_data[0][2], trim = '-')))
#print("cat: " + str(np.format_float_positional(output_data[0][3], trim = '-')))
print("chicken: " + str(np.format_float_positional(output_data[0][3], trim = '-')))
print("classification time: " + str(classification_time) + " seconds")

"""
img = input_data[0]
imgplot = plt.imshow(img/255)
plt.show()
"""