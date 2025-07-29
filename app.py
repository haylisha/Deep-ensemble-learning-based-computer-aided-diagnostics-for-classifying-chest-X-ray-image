import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
import os
import base64
from io import BytesIO
import gdown

# Set page title and layout
st.set_page_config(page_title="Chest X-ray Image Classification", layout="centered")

# Function to convert image to base64 for logo
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Load and display the logo
logo_path = os.path.join(os.getcwd(), "logo.jpeg")
if os.path.exists(logo_path):
    logo_image = Image.open(logo_path)
    logo_base64 = image_to_base64(logo_image)
    st.markdown(
        f"""
        <div style='text-align: center;'>
            <img src='data:image/jpeg;base64,{logo_base64}' width='150'>
        </div>
        """,
        unsafe_allow_html=True
    )

# Add title and description
st.title('Chest X-ray Image Classification')
st.write('This app classifies chest X-ray images into **Normal**, **Pneumonia**, or **Tuberculosis (TB)**.')

# === DOWNLOAD MODEL FROM GOOGLE DRIVE IF NOT PRESENT ===
model_path = 'my_model.h5'
file_id = '1Q1rIjW-3kP0NED8n8vhvRYU4NkkfmYX9'
gdrive_url = f'https://drive.google.com/uc?id={file_id}'

if not os.path.exists(model_path):
    with st.spinner('📥 Downloading model from Google Drive...'):
        gdown.download(gdrive_url, model_path, quiet=False)

# === LOAD MODEL ===
model = tf.keras.models.load_model(model_path)

# === IMAGE PREPROCESSING FUNCTION ===
def preprocess_image(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((150, 150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# === PREDICTION FUNCTION ===
def predict_image(image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)[0]
    class_indices = {0: 'Normal', 1: 'Pneumonia', 2: 'Tuberculosis'}
    predicted_class = class_indices[np.argmax(predictions)]
    predicted_prob = np.max(predictions)
    return predicted_class, predicted_prob, predictions

# === FILE UPLOADER UI ===
uploaded_file = st.file_uploader("📤 Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=False, width=300)

    with col2:
        predicted_class, predicted_prob, all_predictions = predict_image(image)
        st.success(f"Prediction: **{predicted_class}** with **{predicted_prob:.2f}** confidence")
        st.markdown("### All predictions:")
        st.write(f"🫁 **Normal**: {all_predictions[0]:.2f}")
        st.write(f"🌫️ **Pneumonia**: {all_predictions[1]:.2f}")
        st.write(f"🧫 **Tuberculosis**: {all_predictions[2]:.2f}")
