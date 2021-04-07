
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split

def getNumpyArray(dir):
    allImages = []

    for i in os.listdir(dir):
        img = Image.open(dir+"/"+i)
        img = img.resize((100, 100), Image.ANTIALIAS)
        if(np.array(img).shape != (100, 100, 3)):
            continue
        allImages.append(np.array(img).tolist())
    allImages = np.array(allImages)
    return allImages

people = getNumpyArray(r'D:\Downloads\persons')
nothing = getNumpyArray(r'D:\Downloads\none\none')
squirrels = getNumpyArray(r'D:\Downloads\archive\raw-img\squirrel')
cats = getNumpyArray(r'D:\Downloads\archive\raw-img\cat')
chickens = getNumpyArray(r'D:\Downloads\archive\raw-img\chicken')
data = np.concatenate([people, nothing, squirrels, cats, chickens], axis=0)

labels = []
for i in people:
    labels.append([1,0,0,0,0])
for i in nothing:
    labels.append([0,1,0,0,0])
for i in squirrels:
    labels.append([0,0,1,0,0])
for i in cats:
    labels.append([0,0,0,1,0])
for i in chickens:
    labels.append([0,0,0,0,1])
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

np.save(r'D:\Documents\X_train.npy', X_train)
np.save(r'D:\Documents\X_test.npy', X_test)
np.save(r'D:\Documents\y_train.npy', y_train)
np.save(r'D:\Documents\y_test.npy', y_test)