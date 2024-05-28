# Library imports
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from io import BytesIO

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Loading the Model
MODEL = tf.keras.models.load_model("D:\Smart Farmer\Apple\Apple.keras")

# Name of Classes
CLASS_NAMES = ['Black Rot','Cedar Rust','Healthy','Scab']

st.header("Apple Disease Classification")
image = Image.open("D:\Smart Farmer\Apple\Apple.jpg")
# Resize the image
new_size = (375, 260)
resized_image = image.resize(new_size)

# Save the resized image to a new file
resized_image.save("D:\Smart Farmer\Apple\smaller_Apple.jpg")
image = Image.open("D:\Smart Farmer\Apple\smaller_Apple.jpg")
st.image(image, caption='Upload an image of the Data set')


# Uploading the Brain MRI image
Potato_image = st.file_uploader("Choose an image...", type="JPG")
submit = st.button('Predict')

# On predict button click
# On predict button click
if submit:
    if Potato_image is not None:
        # Open the image using PIL
        img = Image.open(Potato_image)

        # Convert the image to RGB if it's not
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize the image to 512x512 pixels.
        img = img.resize((256, 256))

        # Convert the PIL image to a NumPy array
        image = np.array(img)

        # Displaying the resized image
        st.image(image,
                    use_column_width=True)

        # Preparing the image for prediction
        img_batch = np.expand_dims(image, axis=0)

        # Make Prediction
        predictions = MODEL.predict(img_batch)

        # Output prediction
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        if confidence < 0.8:
            st.error("Upload a right image of Apple")
        else:

            st.title(
                f"The Apple has {predicted_class} with confidence {confidence * 100:.2f} %")
