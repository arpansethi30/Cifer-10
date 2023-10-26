import streamlit as st
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier

# Loading the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Flatten the images
x_train_flat = x_train.reshape(x_train.shape[0], -1)
y_train_flat = y_train.ravel()

# Normalize data
scaler = StandardScaler()
x_train_norm = scaler.fit_transform(x_train_flat)

# Train the best model (based on previous results)
model = RandomForestClassifier(n_estimators=50)
model.fit(x_train_norm, y_train_flat)

labels = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

st.title("CIFAR-10 Image Classifier")

uploaded_file = st.file_uploader("Choose a CIFAR-10 like image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess and predict
    image = np.array(image.resize((32, 32)))  # Resize to the input size
    image_flat = image.reshape(1, -1)
    image_norm = scaler.transform(image_flat)

    prediction = model.predict(image_norm)
    st.write(f"Prediction: {labels[int(prediction[0])]}")
