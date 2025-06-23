import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("deepfake_detector.h5")  # otp

model = load_model()


st.title("🧠 Deepfake Image Detector")
st.write("Upload an image, and the model will predict if it's **real** or **fake**.")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0 
    if image.shape[-1] == 4:  
        image = image[:, :, :3]
    image = np.expand_dims(image, axis=0)  
    return image

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]

    if prediction > 0.5:
        st.success(f"✅ Real Image Detected! (Confidence: {prediction:.2f})")
    else:
        st.error(f"🚨 Fake Image Detected! (Confidence: {1 - prediction:.2f})")
