import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


# Attention block
def attention_block(x, g, filters):

    theta_x = Conv2D(filters, (1,1), padding="same")(x)
    phi_g = Conv2D(filters, (1,1), padding="same")(g)

    add = Add()([theta_x, phi_g])
    act = Activation("relu")(add)

    psi = Conv2D(1, (1,1), padding="same")(act)
    psi = Activation("sigmoid")(psi)

    out = Multiply()([x, psi])

    return out


def build_attention_unet(input_shape=(224,224,3)):

    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(64,3,activation='relu',padding='same')(inputs)
    c1 = Conv2D(64,3,activation='relu',padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(128,3,activation='relu',padding='same')(p1)
    c2 = Conv2D(128,3,activation='relu',padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(256,3,activation='relu',padding='same')(p2)
    c3 = Conv2D(256,3,activation='relu',padding='same')(c3)
    p3 = MaxPooling2D()(c3)

    # Bottleneck
    c4 = Conv2D(512,3,activation='relu',padding='same')(p3)
    c4 = Conv2D(512,3,activation='relu',padding='same')(c4)

    # Decoder with Attention
    u5 = UpSampling2D()(c4)
    att3 = attention_block(c3, u5, 256)
    u5 = Concatenate()([u5, att3])

    c5 = Conv2D(256,3,activation='relu',padding='same')(u5)
    c5 = Conv2D(256,3,activation='relu',padding='same')(c5)

    u6 = UpSampling2D()(c5)
    att2 = attention_block(c2, u6, 128)
    u6 = Concatenate()([u6, att2])

    c6 = Conv2D(128,3,activation='relu',padding='same')(u6)
    c6 = Conv2D(128,3,activation='relu',padding='same')(c6)

    u7 = UpSampling2D()(c6)
    att1 = attention_block(c1, u7, 64)
    u7 = Concatenate()([u7, att1])

    c7 = Conv2D(64,3,activation='relu',padding='same')(u7)
    c7 = Conv2D(64,3,activation='relu',padding='same')(c7)

    outputs = Conv2D(1,1,activation="sigmoid")(c7)

    model = Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model