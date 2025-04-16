import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("mushroom_regression_model_9186.keras",compile=False)

# Define function to preprocess image
def preprocess_image(image):
    """Convert and preprocess image for model prediction."""
    try:
        image = Image.open(image)  # Open image (works for both uploaded & captured images)
        image = image.convert("RGB")  # Ensure 3 channels
        image = image.resize((128, 128))  # Resize to match model input
        img_array = np.array(image) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Function to predict mushroom shelf life score
def predict_mushroom(image):
    """Predict shelf life score of the mushroom image."""
    processed_img = preprocess_image(image)
    if processed_img is not None:
        prediction = model.predict(processed_img)[0][0]  # Extract single value
        return round(prediction, 2)
    return None

# Function to get status message
def get_status_message(score):
    """Return freshness status message based on score."""
    if score < 40:
        return "🌱 **They are Fresh! Enjoy your mushrooms!** 🍄"
    elif 40 <= score < 50:
        return "🌿 **Little Spoiled! Still Good for Today!** 🍄"
    elif 50 <= score < 60:
        return "⚠️ **Medium Spoiled! Use with Caution!** ⚠️"
    elif 60 <= score < 70:
        return "❌ **Highly Spoiled! Not Advisable for Eating!** ❌"
    else:
        return "💀 **Rotten! Best Used for Manure!** 🌱"

# Streamlit UI
st.set_page_config(page_title="Mushroom Shelf-Life Predictor", layout="centered")

st.title("🍄 Mushroom Freshness Detector 🍄")
st.write("Upload or capture an image of a mushroom to predict its shelf-life score.")

# Upload or Capture Image
image_source = st.radio("Choose Image Source:", ("Upload Image", "Take a Picture"))

image = None  # Initialize image variable

if image_source == "Upload Image":
    uploaded_file = st.file_uploader("Upload a mushroom image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = uploaded_file  # Use uploaded file directly

elif image_source == "Take a Picture":
    captured_image = st.camera_input("Take a picture of a mushroom")
    if captured_image is not None:
        image = captured_image  # Use captured image directly

# Process the image and predict
if image is not None:
    st.image(image, caption="📸 Uploaded Mushroom Image", use_container_width=True)

    # Prediction Button
    if st.button("🔍 Predict Freshness Score"):
        score = predict_mushroom(image)
        if score is not None:
            status_message = get_status_message(score)

            st.subheader(f"📊 Predicted Shelf Life Score: **{score}**")
            if score < 50:
                st.success(status_message)
            elif score < 70:
                st.warning(status_message)
            else:
                st.error(status_message)
