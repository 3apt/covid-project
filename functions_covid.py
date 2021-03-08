#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 11:12:32 2021

@author: baptiste
"""

import keras
import tensorflow as tf
import numpy as np
from skimage import transform
import streamlit as st
from PIL import Image
import cv2
from skimage import exposure


# algorithme de grad-CAM proposé par Keras : https://keras.io/examples/vision/grad_cam/
def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def decode_predictions(pred):
    if np.argmax(pred) == 1:
        return 'covid'
    elif np.argmax(pred) == 2:
        return 'pneumo'
    else:
        return 'normal'
    
def get_img_array(img, im_shape):
    X = img/255
    X = transform.resize(X, im_shape)
    return X.reshape((1, im_shape[0], im_shape[1], 1))


MODEL = "troisieme_prototype.h5"

@st.cache(allow_output_mutation=True)
def load_model():
    print("chargement du modèle")
    model = keras.models.load_model(f"cropped_dataset/{MODEL}", compile=True)

    return model

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
im_shape = (226, 226)
def preprocess_image(img):
    X = clahe.apply(img)
    X = X/255
    X = exposure.equalize_hist(X)
    X = transform.resize(X, im_shape)
    X = exposure.rescale_intensity(np.squeeze(X), out_range=(0,1))
    return X.reshape((1, im_shape[0], im_shape[1], 1))


def predict(model, img):
    pred = model.predict(img)
    prediction = decode_predictions(pred)

    return np.max(pred), prediction
