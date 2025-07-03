import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="wide"
)

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

model = load_model('checkpoints/mnist_lenet5.h5')

st.title("Handwritten Digit Recognition")
st.write("Draw a digit between 0 and 9, then click predict")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    canvas = st_canvas(
        stroke_width=30,
        background_color="#FFFFFF",
        drawing_mode="freedraw",
        key="canvas"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🔮 Predict")

        if predict_btn and canvas.image_data is not None:
            img = Image.fromarray(canvas.image_data.astype('uint8')).convert('L')
            img = img.resize((28, 28))        
            img_arr = 255 - np.array(img)            
            img_arr = img_arr.reshape(1, 28, 28, 1).astype('float32') / 255.0
                
            pred = model.predict(img_arr)
            pred_digit = np.argmax(pred)

            st.image(img_arr.reshape(28, 28), caption="The model see", width=100)
            st.markdown(f"## Prediction: **{pred_digit}**")
            st.write(f"Confidence: {np.max(pred)*100:.1f}%")
        else:
            st.warning("⚠️ Please draw a digit")