import cv2
import numpy as np
import tensorflow as tf

def gradcam(model, img):

    last_conv = [layer for layer in model.layers if 'conv' in layer.name][-1]

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img)
        class_idx = tf.argmax(preds[0])
        loss = preds[:,class_idx]

    grads = tape.gradient(loss, conv_out)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_out = conv_out[0]

    heatmap = conv_out @ pooled_grads[...,tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)

    return heatmap