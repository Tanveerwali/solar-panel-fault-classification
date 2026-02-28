import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import base64


MODEL_PATH = "solar_fault_model_saved"
IMG_SIZE = (224, 224)
CLASS_NAMES = [
    "Bird_drop",
    "Clean",
    "Dusty",
    "Electrical-Damage",
    "Physical-Damage",
    "Snow-Covered"
]




@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()




def set_background(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{b64}");
        background-size: cover;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


set_background("markus-spiske-pwFr_1SUXRo-unsplash.jpg")




st.set_page_config(page_title="Solar Panel Fault Classifier", layout="wide")


st.markdown("""
    <h1 style='text-align: center; color: #FFD700; font-size: 50px;'>
        🌞 Solar Panel Fault Classifier
    </h1>
    <p style='text-align: center; font-size:20px; color:white;'>
        Upload a solar panel image to classify its condition
    </p>
""", unsafe_allow_html=True)




uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    
    
    img_display = img.resize((400, 300))
    st.image(img_display, caption='Uploaded Image', width=400)

    
    img_resized = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img_resized)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    
    
    
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    
    
    
    st.markdown(f"""
        <h2 style='color: #FFD700; font-size: 35px; font-weight:bold;'>
            ⚡ Prediction: {predicted_class}
        </h2>
        <h3 style='color: #FFD700; font-size:28px;'>
            Confidence: {confidence:.2f}%
        </h3>
    """, unsafe_allow_html=True)

    
    
    
    st.markdown("""
        <style>
        .stProgress > div > div > div > div {
            background-color: #00FF00 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    st.progress(int(confidence))

    
    
    
    result_text = f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%"
    st.download_button("📥 Download Result", result_text, file_name="prediction.txt")