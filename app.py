import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import os

# Class names
class_names = ['butterfly', 'cat', 'chicken', 'cow',
               'dog', 'elephant', 'horse', 'sheep',
               'spider', 'squirrel']

# Animal Info
animal_info = {
    'dog': '🐶 Dog — Insaan ka sabse wafadar dost!',
    'cat': '🐱 Cat — Akela rehna pasand karta hai!',
    'horse': '🐴 Horse — Sabse tez dauda karta hai!',
    'elephant': '🐘 Elephant — Sabse badi yadasht!',
    'butterfly': '🦋 Butterfly — Phoolon ka dost!',
    'chicken': '🐔 Chicken — Subah jagata hai!',
    'cow': '🐄 Cow — Doodh deti hai!',
    'sheep': '🐑 Sheep — Unn deta hai!',
    'spider': '🕷️ Spider — Jaal bunata hai!',
    'squirrel': '🐿️ Squirrel — Akhrot ka shaukeen!'
}

# Page setup
st.set_page_config(
    page_title="Animal Species Identifier",
    page_icon="🐾",
    layout="centered"
)

# Model download karo Google Drive se
@st.cache_resource
def load_my_model():
    model_path = "animal_classifier.h5"
    if not os.path.exists(model_path):
        with st.spinner("🔄 AI Model load ho raha hai..."):
            gdown.download(
                f"https://drive.google.com/uc?id=1RYS-nb-EVaXVhBD7UgN8DexBrNDljMJQ",
                model_path,
                quiet=False
            )
    return load_model(model_path)

# Title
st.title("🐾 Animal Species Identifier")
st.write("Kisi bhi animal ki photo upload karo — AI bata dega!")
st.divider()

model = load_my_model()

# Image upload
uploaded_file = st.file_uploader(
    "📸 Animal ki image upload karo",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=300)

    if st.button("🔍 Identify Animal!"):
        with st.spinner("AI soch raha hai..."):
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions)]
            confidence = round(100 * np.max(predictions), 2)

        st.divider()
        st.success(f"✅ Ye hai: **{predicted_class.upper()}**")
        st.info(f"📊 Confidence: **{confidence}%**")
        st.write(animal_info[predicted_class])
        st.progress(int(confidence))

st.divider()
st.caption("Made with ❤️ using TensorFlow & Streamlit")