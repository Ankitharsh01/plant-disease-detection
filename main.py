import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Load model and class indices
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Set custom styles (background, fonts, buttons)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    .reportview-container {
        background: url("galgotias.jpg");
        background-size: cover;
        background-attachment: fixed;
        color: white;
        font-family: 'Roboto', sans-serif;
    }
    .title {
        font-size: 2.5em;
        font-weight: 700;
        margin-bottom: 20px;
        text-shadow: 2px 2px 5px black;
    }
    .header {
        font-size: 1.75em;
        font-weight: 600;
        color: #f4f4f4;
        text-shadow: 1px 1px 3px black;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
        box-shadow: 0px 5px 10px rgba(0,0,0,0.2);
    }
    .footer {
        text-align: center;
        color: #f4f4f4;
        padding: 10px;
        margin-top: 50px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True
)

# Title and description
st.markdown('<h1 class="title">🌱 Plant Disease Classifier 🌿</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="header">Identify plant diseases from images using deep learning</h2>', unsafe_allow_html=True)

# Display galgotias.jpg image at the top
st.image("galgotias.jpg", caption="Galgotias University", use_column_width=True)
# Add logo with specific dimensions (e.g., width and height)
#st.image("galgotias.jpg", caption="Galgotias University",width=500, height=300)  # Set width and height for the image


# Image uploader section
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Image display and prediction
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Process and predict
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')

# Footer with credits
st.markdown(
    """
    <div class='footer'>
        Made with ❤️ by Ankit Harsh and Mohit Kumar
    </div>
    """, unsafe_allow_html=True
)
