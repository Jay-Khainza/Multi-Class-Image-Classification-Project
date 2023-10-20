import streamlit as st
import tensorflow as tf
import os
from PIL import Image
import numpy as np

# Load the saved CNN model
model = tf.keras.models.load_model('model3.h5')

# Define class labels
class_labels = ['actinic keratosis',
 'basal cell carcinoma',
 'dermatofibroma',
 'melanoma',
 'nevus',
 'pigmented benign keratosis',
 'seborrheic keratosis',
 'squamous cell carcinoma',
 'vascular lesion']

# Streamlit app title and description
st.title("Skin Condition Image Classifier")
st.write("This app classifies images of skin conditions into one of nine classes.")

# File upload widget
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Image preprocessing
    image = Image.open(uploaded_image)
    image = image.resize((64, 64))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(image)
    class_index = np.argmax(predictions)

    # Display the predicted class
    st.write(f"Prediction: {class_labels[class_index]}")

# Footer
st.write("This is a simple skin condition image classification app using a CNN model.")
