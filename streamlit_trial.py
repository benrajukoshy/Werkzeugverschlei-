import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# loading the necessary libraries
import numpy as np
import os
import random
#import cv2
import PIL.Image as Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from keras.utils import to_categorical
#from google.colab.patches import cv2_imshow
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from sklearn.model_selection import train_test_split

# Load and preprocess the data
import pandas as pd
import random
from urllib.request import urlopen, urlretrieve
from PIL import Image
#from tqdm import tqdm_notebook
from sklearn.utils import shuffle
#import cv2
from sklearn.metrics import precision_recall_curve

# Import the pre-trained models
from tensorflow.keras.applications import ResNet50, MobileNetV2

# Define callbacks and training parameters
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint

# Build the model
from keras.models import Sequential, Model, load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D

# Funktion zum Laden des Skalierers aus der Pickle-Datei
# def load_scaler(filename):
#     with open(filename, 'rb') as file:
#         scaler = pickle.load(file)
    # return scaler
print(tf.__version__)
# Laden des trainierten Modells
filename = '/workspaces/Werkzeugverschlei-/mobilenet_v2_version1 (1).keras'
model = keras.models.load_model(filename)

# Laden der Skalierer
scaler = load_scaler('ScaleFaktors_X.sav')
scaler_y = load_scaler('ScaleFaktors_y.sav')

def scale_input(input_values, scaler):
    X_scaled = scaler.transform(np.array([input_values]))
    return X_scaled
    
def inverse_scale_output(output, scaler_y):
    return scaler_y.inverse_transform(np.array([output]).reshape(-1, 1))

def main():
    st.title("Regressionsübung im ML Seminar, WS23/24")
    st.header("Prognose der Betonfestigkeit")

    # Abschnitt für SelectSlider-Elemente
    st.header("Wählen Sie die Mengen Ihrer Betoninhaltsstoffe aus")

    # Variablen und ihre Bereichsgrenzen
    variables = {
        "cement": (100, 500),
        "slag": (0, 200),
        "flyash": (0, 200),
        "water": (100, 300),
        "superplasticizer": (0, 30),
        "coarseaggregate": (800, 1200),
        "fineaggregate": (600, 1000),
        "age": (1, 365)
    }


    values = []
    for var, (min_val, max_val) in variables.items():
        value = st.select_slider(f"{var.capitalize()} (Einheit)", range(min_val, max_val + 1))
        values.append(value)

    if st.button("Vorhersage machen"):
        input_values_scaled = scale_input(values, scaler)
        prediction_scaled = model.predict(input_values_scaled)
        prediction = inverse_scale_output(prediction_scaled, scaler_y)
        st.write("Prognostizierte Festigkeit Ihres Betons in MPa:")
        st.text_area("Ergebnis", f"{prediction}", height=100)

if __name__ == "__main__":
    main()
