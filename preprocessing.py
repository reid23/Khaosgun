# %%
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from numba import njit
# %%


def getNumpyArray(dir, imageShape=(128, 128, 3), pprocess=['MONOCOLOR', 'GAUSSIAN', 'EDGES']):
    allImages = []
    for i in os.listdir(dir):
        counter = 0
        img = cv2.imread(dir+"/"+i)
        img = cv2.resize(img, (imageShape[0], imageShape[1]))
        if(img.shape != imageShape):
            print("image " + dir + "/" + i + " has shape of " + str(list(np.array(img).shape)) +
                  ", which is incompatible with the asserted output shape.  This image will be ignored.")
            continue
        for i in pprocess:
            if(i == 'EDGES'):
                x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=7)
                y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=7)
                img = cv2.addWeighted(x, 0.5, y, 0.5, 0)
            elif(i == 'GAUSSIAN'):
                img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
            elif(i == 'MONOCOLOR'):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            counter += 1
        allImages.append(np.array(img).tolist())
    allImages = np.array(allImages)
    print(dir + " loaded to numpy array")
    return allImages


# %%
arrays = getNumpyArray('/home/reid/projects/khaosPhotos/person',
                       (128, 128, 3), pprocess=['MONOCOLOR'])
for i in arrays:
    plt.show(plt.imshow(i, cmap='gray'))

# %%
people = getNumpyArray(r'/home/reid/projects/khaosPhotos/person',
                       (128, 128, 3), pprocess=['MONOCOLOR', 'GAUSSIAN'])
nothing = getNumpyArray(r'/home/reid/projects/khaosPhotos/none',
                        (128, 128, 3), pprocess=['MONOCOLOR', 'GAUSSIAN'])
data = np.concatenate([people, nothing], axis=0)

print("Data processed")

labels = []
for i in people:
    labels.append([1, 0])
print("Labels made for category 'people'")
for i in nothing:
    labels.append([0, 1])
print("Labels made for category 'nothing'")

labels = np.array(labels)

print("Labels processed")

print("data shape: " + str(list(data.shape)))
print("labels shape: " + str(list(labels.shape)))

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=53)

print("processed shapes:")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_test[1:10])
print(y_train[1:10])

np.save(r'/home/reid/Documents/khaosTrainData/X_train1', X_train)
np.save(r'/home/reid/Documents/khaosTrainData/X_test1', X_test)
np.save(r'/home/reid/Documents/khaosTrainData/y_train1', y_train)
np.save(r'/home/reid/Documents/khaosTrainData/y_test1', y_test)

# %%
