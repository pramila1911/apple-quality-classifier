
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Title
st.title("🍎 Apple Quality Classifier")
st.write("Upload an apple image to check if it's Good or Defective.")

# Load the model
model = load_model("apple_model.h5")

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    result = "✅ Good Apple" if prediction > 0.5 else "❌ Defective Apple"
    st.markdown(f"### Prediction: {result}")
