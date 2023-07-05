# -*- coding: utf-8 -*-
"""HED.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sxAQYKOi_hJozIVNX80ChnznWExGtWXd

# IMPORTS
"""

import tensorflow as tf # For tensorflow
import numpy as np # For mathematical computations
import matplotlib.pyplot as plt # For plotting and Visualization
import seaborn as sns
from tensorflow.keras.layers import Input, Layer, Resizing, Rescaling, InputLayer, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, RandomRotation, RandomFlip, RandomContrast, ReLU, Add, GlobalAveragePooling2D, Permute
from tensorflow.keras import Model
from tensorflow.keras.regularizers import L2
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import cv2

"""# ONNX"""

!pip install onnx
!pip install onnxruntime

import onnx
import onnxruntime as ort

# from google.colab import drive
# drive.mount('/content/drive')



"""## Predicting Using Onnx Model"""

!pip install onnx
!pip install onnxruntime

import onnxruntime as rt
import onnx

"""# Creating Web Interface Using Gradio"""

!pip install gradio

import gradio as gr

!pip install onnx
!pip install onnxruntime

import onnxruntime as rt

# !cp -r /content/drive/MyDrive/vit_onnx.onnx /content/vit_onnx.onnx

import onnx
model = onnx.load("vit_onnx.onnx")

import onnxruntime as ort
session = ort.InferenceSession("vit_onnx.onnx")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

CLASS_NAMES = ["Angry", "Happy", "Sad"]
def predict_image(im):
  im = tf.expand_dims(tf.cast(im, tf.float32), axis=0).numpy()
  prediction = session.run([output_name], {input_name: im})
  return {CLASS_NAMES[i]: float(prediction[0][0][i]) for i in range(3)}

image = gr.inputs.Image(shape=(224, 224))
label = gr.outputs.Label(num_top_classes=3)
iface = gr.Interface(fn=predict_image, inputs=image, outputs=label, capture_session=True)
iface.launch(debug="True")

