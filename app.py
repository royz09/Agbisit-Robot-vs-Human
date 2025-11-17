import streamlit as st
import numpy as np
from PIL import Image
import time
import json
import tensorflow as tf

# Set page configuration
st.set_page_config(
    page_title="Robot vs Human Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .human-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        border: 3px solid #4ECDC4;
    }
    .robot-card {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        border: 3px solid #FFD93D;
    }
    .prediction-text {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        text-align: center;
    }
    .confidence-text {
        font-size: 1.5rem;
        text-align: center;
    }
    .confidence-bar {
        background: rgba(255,255,255,0.3);
        border-radius: 10px;
        margin: 15px 0;
        overflow: hidden;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #FFD93D, #FF6B6B);
        height: 30px;
        border-radius: 10px;
        text-align: center;
        color: black;
        font-weight: bold;
        line-height: 30px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------
# LOAD MODEL
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
        with open('class_info.json', 'r') as f:
            return json.load(f)
    except:
        return {
            "Robot": {
                "description": "Mechanical or electronic beings with artificial intelligence",
                "characteristics": [
                    "Metallic surfaces and mechanical parts",
                    "LED lights or electronic components",
                    "Angular and geometric shapes",
                    "Wires, circuits, or robotic joints",
                    "Artificial appearance"
                ],
                "examples": [
                    "Industrial robots",
                    "Humanoid robots",
                    "Sci-fi androids",
                    "Toy robots",
                    "AI assistants"
                ]
            },
            "Human": {
                "description": "Organic beings with natural biological features",
                "characteristics": [
                    "Skin tones and organic textures",
                    "Facial features and expressions",
                    "Hair and natural colors",
                    "Clothing and accessories",
                    "Natural body proportions"
                ],
                "examples": [
                    "People in photographs",
                    "Portraits and selfies",
                    "Human characters in art",
                    "Human figures in drawings"
                ]
            }
        }


# ------------------------------
# IMAGE PREPROCESSING
# ------------------------------
def preprocess_image(image):
    """Resize, normalize, and batch the image."""
    image = image.resize((128, 128))
    img_array = np.array(image)

    # Ensure 3 channels
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ------------------------------
# REAL MODEL PREDICTION
# ------------------------------
def predict_image(model, image):
    processed = preprocess_image(image)

    with st.spinner("🔍 Analyzing image using CNN..."):
        time.sleep(1)
        prediction = model.predict(processed)[0][0]  # Assuming binary output

    # Model output: 0 = Human, 1 = Robot  (update if reversed)
    robot_conf = float(prediction)
    human_conf = 1 - robot_conf

    return human_conf, robot_conf


# ------------------------------
# MAIN APP
# ------------------------------
def main():
    st.markdown('<h1 class="main-header">🤖 Robot vs Human Classifier</h1>', unsafe_allow_html=True)

    st.write("""
    ### Deep Learning Image Classification  
    Upload an image and let the model classify it as **Robot** or **Human**!
    """)

    model = load_model()
    class_info = load_class_info()

    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.info("""
        **Binary Classifier**  
        - Human (0)  
        - Robot (1)  
        """)

        st.header("⚙️ CNN Details")
        st.write("Input: 128×128")
        st.write("Framework: TensorFlow/Keras")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📸 Upload Image")
        uploaded_file = st.file_uploader(
            "Upload a picture of a Robot or Human",
            type=["jpg", "jpeg", "png"]
        )

        image = None
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=400)

    with col2:
        st.subheader("🔍 Classification Guide")
        st.write("Robots → metallic, mechanical, LEDs")
        st.write("Humans → skin tone, hair, natural features")

    # Prediction
    if image is not None:
        st.markdown("---")
        st.subheader("🎯 Classification Results")

        human_conf, robot_conf = predict_image(model, image)

        if human_conf > robot_conf:
            prediction = "Human"
            confidence = human_conf
            card = "human-card"
            emoji = "👤"
        else:
            prediction = "Robot"
            confidence = robot_conf
            card = "robot-card"
            emoji = "🤖"

        st.markdown(f"""
        <div class="{card}">
            <div class="prediction-text">{emoji} {prediction}</div>
            <div class="confidence-text">Confidence: {confidence:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bars
        st.subheader("📊 Confidence Levels")

        # Robot %
        st.write("🤖 Robot Confidence")
        st.markdown(f"""
        <div class="confidence-bar">
            <div class="confidence-fill" style="width:{robot_conf*100}%;">{robot_conf:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

        # Human %
        st.write("👤 Human Confidence")
        st.markdown(f"""
        <div class="confidence-bar">
            <div class="confidence-fill" style="width:{human_conf*100}%;">{human_conf:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

        # Additional class info
        with st.expander("📖 Class Information"):
            info = class_info[prediction]
            st.write(f"### {prediction} Characteristics")
            st.write(info["description"])
            st.write("**Common traits:**")
            for item in info["characteristics"]:
                st.write("-", item)


if __name__ == "__main__":
    main()
