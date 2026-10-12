import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

IMG_SIZE = 224

X, Y = [], []

img_dir = "dataset/segmentation/images"
mask_dir = "dataset/segmentation/masks"

for file in os.listdir(img_dir):

    img = cv2.imread(os.path.join(img_dir, file))
    mask = cv2.imread(os.path.join(mask_dir, file), 0)

    if img is None or mask is None:
        continue

    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE)) / 255.0
    mask = cv2.resize(mask,(IMG_SIZE,IMG_SIZE))
    mask = (mask > 127).astype(np.float32)
    mask = np.expand_dims(mask, -1)

    X.append(img)
    Y.append(mask)

X = np.array(X)
Y = np.array(Y)

inputs = Input((IMG_SIZE,IMG_SIZE,3))

c1 = Conv2D(32,3,activation='relu',padding='same')(inputs)
p1 = MaxPooling2D()(c1)

c2 = Conv2D(64,3,activation='relu',padding='same')(p1)
p2 = MaxPooling2D()(c2)

c3 = Conv2D(128,3,activation='relu',padding='same')(p2)

u1 = UpSampling2D()(c3)
c4 = Conv2D(64,3,activation='relu',padding='same')(u1)

u2 = UpSampling2D()(c4)
c5 = Conv2D(32,3,activation='relu',padding='same')(u2)

output = Conv2D(1,1,activation='sigmoid')(c5)

model = Model(inputs, output)

model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(X, Y, epochs=15)

model.save("models/segmentation_model.h5")

print("✅ Segmentation ready")