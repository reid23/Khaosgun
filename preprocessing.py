
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split


def getNumpyArray(dir,imageShape=(256,256,3)):
    allImages = []

    for i in os.listdir(dir):
        img = Image.open(dir+"\\"+i)
        img = img.resize((imageShape[0], imageShape[1]), Image.ANTIALIAS)
        allImages.append(np.array(img).tolist())
        print("beep boop! " + i)
    allImages = np.array(allImages)
    print(dir + " loaded to numpy array")
    return allImages


people = getNumpyArray(r'D:\Downloads\persons')
nothing = getNumpyArray(r'D:\Downloads\none\none')
squirrels = getNumpyArray(r'D:\Downloads\archive\raw-img\squirrel')
#cats = getNumpyArray(r'D:\Downloads\archive\raw-img\cat')
chickens = getNumpyArray(r'D:\Downloads\archive\raw-img\chicken')
data = np.concatenate([people, nothing, squirrels, chickens], axis=0)

print("Data processed")

labels = []
for i in people:
    labels.append([1,0,0,0])
print("Labels made for category 'people'")
for i in nothing:
    labels.append([0,1,0,0])
print("Labels made for category 'nothing'")
for i in squirrels:
    labels.append([0,0,1,0])
print("Labels made for category 'squirrels'")
#for i in cats:
#    labels.append([0,0,0,1,0])
print("Labels made for category 'cats'")
for i in chickens:
    labels.append([0,0,0,1])
print("Labels made for category 'chickens'")
labels = np.array(labels)

print("Labels processed")

print("data shape: " + str(list(data.shape)))
print("labels shape: " + str(list(labels.shape)))

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print("processed shapes:")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

np.save(r'D:\Documents\X_train', X_train)
np.save(r'D:\Documents\X_test', X_test)
np.save(r'D:\Documents\y_train', y_train)
np.save(r'D:\Documents\y_test', y_test)