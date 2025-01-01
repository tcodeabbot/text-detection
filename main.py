

import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

# Streamlit app
st.title("Text Detection App")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    img = np.array(image)

    # Instance text detector
    reader = easyocr.Reader(['en'], gpu=False)

    # Detect text on image
    text_ = reader.readtext(img)

    threshold = 0.25
    # Draw bbox and text
    for t_, t in enumerate(text_):
        print(t)

        bbox, text, score = t

        if score > threshold:
            cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 5)
            cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

    # Display image with detected text
    st.image(img, caption='Processed Image', use_container_width=True, clamp=True)