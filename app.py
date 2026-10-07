import os
import json
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_disease_model.keras")
CLASS_PATH = os.path.join(BASE_DIR, "class_names.json")
IMG_SIZE = (128, 128)

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿")
st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image and the CNN model will predict its class.")

if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_PATH):
    st.error("Model not found. Please run train.py first.")
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
class_names = load_classes()

file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded leaf", use_container_width=True)

    arr = np.array(image.resize(IMG_SIZE), dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    probs = model.predict(arr, verbose=0)[0]

    idx = int(np.argmax(probs))
    st.success(f"Prediction: {class_names[idx]}")
    st.write(f"Confidence: {probs[idx] * 100:.2f}%")

    st.subheader("All class probabilities")
    st.bar_chart({class_names[i]: float(probs[i]) * 100 for i in range(len(class_names))})
