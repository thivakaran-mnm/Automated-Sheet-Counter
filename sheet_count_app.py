import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load trained model
try:
    model = load_model(r'C:\Users\hp\sheet_count_model_1.keras')  # Update the path to your saved model
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Image preprocessing function
def preprocess_image(image):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        image = cv2.resize(image, (224, 224))  # Resize to match model input
        image = cv2.Canny(image, 100, 200)  # Applying edge detection
        image = image / 255.0  # Normalize
        image = np.expand_dims(image, axis=-1)  # Adding channel dimension
        image = np.expand_dims(image, axis=0)  # Adding batch dimension
        return image
    except Exception as e:
        st.error(f"Error in preprocessing image: {e}")
        st.stop()

# Streamlit UI
st.set_page_config(page_title="Sheet Counter", page_icon="📝")
st.markdown("""
    <style>
    .main {
        background-color: #f0faff;  
        color: #333;
        font-family: 'Arial', sans-serif;
    }
    .title {
        text-align: center;
        color: #1E90FF;  
        font-size: 2.5em;
        padding: 20px;
    }
    .caption {
        text-align: center;
        color: #4682B4;  
        font-size: 1.5em;
    }
    .button {
        background-color: #1E90FF;  
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 10px 2px;
        cursor: pointer;
        border-radius: 5px;
    }
    .instructions {
        background-color: #e6f7ff;  
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">_-_SHEET COUNTER_-_</h1>', unsafe_allow_html=True)

# File uploader widget
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner('Processing your image...'):
        try:
            # Read the image
            image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            # Preprocess the image
            preprocessed_image = preprocess_image(image)

            # Make prediction
            prediction = model.predict(preprocessed_image)
            count = int(prediction[0][0])  

            # Creating columns for side-by-side display
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption='Uploaded Image', width=150)
            
            with col2:
                st.markdown("")
                st.markdown(f'<h2 class="caption">Estimated number of sheets: </h2>', unsafe_allow_html=True)
                st.markdown(f'<h1 class="title"><span style="color:#1E90FF;">{count}</span></h1>', unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error during image processing or prediction: {e}")

# Instructions 
st.markdown("""
    <div class="instructions">
        <span style="color:#1E90FF;">
        <h4 style="color: #4682B4;">How to Use This App:</h4>
        <p>1. Upload an image of the sheets you want to count.</p>
        <p>2. The app will process the image and display the estimated number of sheets.</p>
        </span>
    </div>
""", unsafe_allow_html=True)
