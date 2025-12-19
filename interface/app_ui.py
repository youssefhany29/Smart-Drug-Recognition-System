import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

from src.config import MODELS_DIR, IMG_SIZE

st.set_page_config(page_title="Smart Drug Recognition", page_icon="💊", layout="centered")

st.title("💊 Smart Drug Recognition")
st.write("Upload a drug package image to classify it as **Original** or **Counterfeit**.")

MODEL_PATH = MODELS_DIR / "smart_drug_model.keras"  

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0) 
    return arr

model = load_model()

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    x = preprocess_image(image)
    prob_original = float(model.predict(x, verbose=0)[0][0]) 

    if prob_original >= 0.5:
        label = "Original"
        confidence = prob_original
    else:
        label = "Counterfeit"
        confidence = 1 - prob_original

    st.subheader("Prediction")
    st.write(f"**Result:** {label}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")

    st.caption("Note: Confidence is based on the sigmoid output. Threshold = 0.5.")

