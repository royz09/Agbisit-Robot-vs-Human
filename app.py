import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json
import time

# Streamlit page config
st.set_page_config(
    page_title="Robot vs Human Classifier",
    page_icon="🤖",
    layout="wide"
)

# ------------------------------
# LOAD MODEL SAFELY
# ------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model


# ------------------------------
# LOAD CLASS INFO
# ------------------------------
def load_class_info():
    try:
        with open("class_info.json", "r") as f:
            return json.load(f)
    except:
        return {
            "Robot": {
                "description": "Mechanical or electronic entities with artificial intelligence.",
                "characteristics": [
                    "Metallic surfaces",
                    "LED lights / glowing eyes",
                    "Mechanical joints",
                    "Robotic limbs",
                    "Wires and circuits"
                ],
                "examples": [
                    "Industrial robots",
                    "Humanoid robots",
                    "Sci-fi androids"
                ]
            },
            "Human": {
                "description": "Biological organisms with natural body features.",
                "characteristics": [
                    "Skin tones",
                    "Human facial features",
                    "Hair",
                    "Natural proportion limbs",
                    "Clothing textures"
                ],
                "examples": [
                    "Portrait photos",
                    "Selfies",
                    "People in artworks"
                ]
            }
        }


# ------------------------------
# IMAGE PREPROCESSING
# ------------------------------
def preprocess_image(img: Image.Image):
    img = img.resize((128, 128))
    arr = np.array(img)

    # Ensure RGB
    if len(arr.shape) == 2:
        arr = np.stack([arr]*3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[:, :, :3]

    arr = arr.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# ------------------------------
# MODEL PREDICTION
# ------------------------------
def predict(model, img):
    processed = preprocess_image(img)

    with st.spinner("Analyzing image..."):
        time.sleep(1)
        pred = model.predict(processed)[0][0]

    # Assumption:
    # 1 = Robot
    # 0 = Human
    robot_conf = float(pred)
    human_conf = 1 - robot_conf

    return human_conf, robot_conf


# ------------------------------
# MAIN APP UI
# ------------------------------
def main():
    st.markdown(
        "<h1 style='text-align:center;'>🤖 Robot vs Human Classifier</h1>",
        unsafe_allow_html=True,
    )

    st.write("Upload an image and the TensorFlow CNN will classify it as **Human** or **Robot**.")

    model = load_model()
    class_info = load_class_info()

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        img = None
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded Image", width=350)

    # SIDE INFO
    with col2:
        st.subheader("Model Information")
        st.info("Trained CNN model (.h5) using TensorFlow 2.13\n128×128 RGB Input\nBinary: Robot (1) vs Human (0)")

        st.subheader("Feature Guide")
        st.write("**Robots:** Metal, LEDs, mechanical parts")
        st.write("**Humans:** Skin tone, hair, natural face")

    # Prediction Section
    if img:
        st.markdown("---")
        st.subheader("Classification Result")

        human_conf, robot_conf = predict(model, img)

        if human_conf > robot_conf:
            label = "Human"
            emoji = "👤"
            conf = human_conf
        else:
            label = "Robot"
            emoji = "🤖"
            conf = robot_conf

        st.success(f"### {emoji} Prediction: **{label}** (Confidence: {conf:.2%})")

        st.write("### Confidence Breakdown")
        st.write(f"🤖 Robot: **{robot_conf:.2%}**")
        st.progress(robot_conf)

        st.write(f"👤 Human: **{human_conf:.2%}**")
        st.progress(human_conf)

        with st.expander("Class Information"):
            info = class_info[label]
            st.write(f"### {label}")
            st.write(info["description"])
            st.write("**Characteristics:**")
            for c in info["characteristics"]:
                st.write("•", c)
            st.write("**Examples:**")
            for e in info["examples"]:
                st.write("•", e)


if __name__ == "__main__":
    main()
