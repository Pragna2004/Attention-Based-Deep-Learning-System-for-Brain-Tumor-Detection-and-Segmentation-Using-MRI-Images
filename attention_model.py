import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def attention_block(x):

    gap = GlobalAveragePooling2D()(x)
    dense1 = Dense(x.shape[-1], activation="relu")(gap)
    dense2 = Dense(x.shape[-1], activation="sigmoid")(dense1)

    scale = Reshape((1,1,x.shape[-1]))(dense2)

    return Multiply()([x, scale])


def build_model(input_shape=(224,224,3), classes=4):

    inputs = Input(shape=input_shape)

    x = Conv2D(32,(3,3),activation="relu")(inputs)
    x = MaxPooling2D()(x)

    x = Conv2D(64,(3,3),activation="relu")(x)
    x = MaxPooling2D()(x)

    x = attention_block(x)

    x = Conv2D(128,(3,3),activation="relu",name="last_conv")(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(256,activation="relu")(x)

    outputs = Dense(classes,activation="softmax")(x)

    model = Model(inputs,outputs)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model