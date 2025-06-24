
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load the trained model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

st.set_page_config(page_title="MNIST Digit Classifier")
st.title("🧠 MNIST Digit Classifier")
st.write("Upload a 28x28 grayscale image of a handwritten digit (0–9) and get the model's prediction.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    st.image(image, caption="Processed Image", width=150)

    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    st.subheader(f"🧾 Predicted Digit: {predicted_digit}")
