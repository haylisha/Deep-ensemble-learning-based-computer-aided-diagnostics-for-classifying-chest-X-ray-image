import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
import os
import base64
from io import BytesIO

# Set page title and center content
st.set_page_config(page_title="Chest X-ray Image Classification", layout="centered")

# Function to convert image to base64
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Load and center the logo
logo_path = os.path.join(os.getcwd(), "logo.jpeg")  # Assuming logo.png is in the same directory as your script
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

# Add title
st.title('Chest X-ray Image Classification')

# Add description
st.write('This app classifies chest X-ray images into Normal, Pneumonia, or Tuberculosis (TB)')

# Load the trained model
model_path = 'my_model.h5'  # If my_model.h5 is in the same folder as this script
model = tf.keras.models.load_model(model_path)

def preprocess_image(img):
    # Convert image to RGB mode if it's not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Resize the image to (150, 150)
    img = img.resize((150, 150))
    # Convert the image to a numpy array and normalize
    img = np.array(img)
    img = img / 255.0
    # Add a batch dimension
    img = np.expand_dims(img, axis=0)
    return img

# Function to make predictions
def predict_image(image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)[0]
    class_indices = {0: 'Normal', 1: 'Pneumonia', 2: 'Tuberculosis'}
    predicted_class = class_indices[np.argmax(predictions)]
    predicted_prob = np.max(predictions)
    return predicted_class, predicted_prob, predictions

# File uploader
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=False, width=300)

    with col2:
        # Make a prediction and display the result
        predicted_class, predicted_prob, all_predictions = predict_image(image)
        st.write(f"Prediction: **{predicted_class}** with **{predicted_prob:.2f}** probability")
        st.write(f"**All predictions:**")
        st.write(f"Normal: {all_predictions[0]:.2f}")
        st.write(f"Pneumonia: {all_predictions[1]:.2f}")
        st.write(f"Tuberculosis: {all_predictions[2]:.2f}")
