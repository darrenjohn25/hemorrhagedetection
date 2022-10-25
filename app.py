import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras as kp
import numpy as np
from PIL import Image

# new branch changes
st.markdown("<h1 style='text-align: center;'>Hemorrhage Detection</h1>", unsafe_allow_html=True)
#change in master
model = kp.models.load_model("weightsVGG.h5")
### load file
file = st.file_uploader("Upload a CT file of Brain in Jpeg")
if file is not None:
    image = Image.open(file)
    st.image(
        image,
    )
    image = np.array(image)
    image = image[...,::-1]
    img = tf.image.resize(image, size=(224,224))
    img = np.array(img / 255.0)
    image_batch = np.expand_dims(img,axis=0)
    prediction = model.predict(image_batch)
    pred_new= np.argmax(prediction, axis=1)
    if (pred_new == 1):
        st.title("Predicted Label for the image is Detected Hemorrhage")
    else:
        st.title("Predicted Label for the image is Normal")