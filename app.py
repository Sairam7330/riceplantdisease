import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

# Google Drive file ID
file_id = "1SaDGf-5F74No2MvgJL83P-1POoBFJp1n"
output_path = "/mnt/data/rice_plant_disease_model.h5"

@st.cache_resource()
def download_and_load_model():
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
    model = tf.keras.models.load_model(output_path)
    return model

model = download_and_load_model()

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Rice Disease Detection")
st.write("Upload an image of a rice plant leaf to detect disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_image)
    
    # Updated class labels based on the provided order
    class_labels = [
        "Hispa",
        "Bacterial Leaf Blight",
        "Shath Blight",
        "Leaf Scald",
        "Leaf Blast",
        "Healthy",
        "Narrow Brown Spot",
        "Brown Spot",
        "Tungro"
    ]
    
    predicted_class = class_labels[np.argmax(prediction)]
    
    st.write(f"### Predicted Disease: {predicted_class}")
