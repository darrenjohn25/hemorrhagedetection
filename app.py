import numpy as np
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# new branch changes
st.markdown("<h1 style='text-align: center;'>Hemorrhage Detection</h1>", unsafe_allow_html=True)
#change in master
model = tf.keras.models.load_model("weightsVGG.h5")
### load file
file = st.file_uploader("Upload a CT file of Brain in Jpeg")
if file is not None:
    image = Image.open(file)
    st.image(
        image,
    )
    image = np.array(image)
    # img=tf.image.grayscale_to_rgb(image)
    image = image[...,::-1]
    img = tf.image.resize(image, size=(224,224))
    img = np.array(img / 255.0)
    prediction = model.predict(img)
    pred_new= np.argmax(prediction, axis=1)
    if (pred_new == 1):
        st.title("Predicted Label for the image is Normal")
    else:
        st.title("Predicted Label for the image is Covid-19")