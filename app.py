import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load trained model
model = load_model('brain_tumor_model.h5')

# Set page title
st.title("🧠 Brain Tumor Detection from MRI Scan")

# Sidebar with instructions and tech stack
st.sidebar.title("📌 Instructions")
st.sidebar.info(
    "1. Upload an MRI image (JPG, JPEG, PNG).\n"
    "2. Click the 'Predict Tumor' button to see results."
)

st.sidebar.title("🛠 Tech Stack")
st.sidebar.markdown(
    """
    - Python 🐍  
    - Streamlit 📊  
    - TensorFlow 🤖  
    - OpenCV & PIL 🖼️
    """
)

# File uploader
uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded MRI Image", use_column_width=True)

    # Button to trigger prediction
    if st.button("🧪 Predict Tumor"):
        # Preprocess the image using Keras
        img = img.resize((150, 150))
        img_array = img_to_array(img)
        
        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]  # Remove alpha channel if present

        img_array = np.expand_dims(img_array, axis=0)  # (1, 150, 150, 3)
        img_array = img_array / 255.0  # Normalize

        # Make prediction
        prediction = model.predict(img_array)[0][0]
        result = "🛑 Positive (Tumor Detected)" if prediction > 0.5 else "✅ Negative (No Tumor)"

        # Display result
        st.subheader("🔍 Prediction Result:")
        st.success(result)

