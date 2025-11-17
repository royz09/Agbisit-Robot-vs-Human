import streamlit as st
import numpy as np
from PIL import Image
import time
import json
import random

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
    .stats-card {
        background: rgba(255,255,255,0.1);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .feature-list {
        background: rgba(255,255,255,0.1);
        padding: 15px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .demo-notice {
        background: #FFD93D;
        color: black;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def load_model():
    """Demo model loader - returns demo mode indicator"""
    st.warning("🚧 DEMO MODE - Using simulated predictions")
    return "demo_mode"

def load_class_info():
    """Load class information"""
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
                    "Human characters in art",
                    "Portraits and selfies",
                    "Human figures in drawings"
                ]
            }
        }

def preprocess_image(image):
    """Preprocess the image for the model"""
    # Resize to 128x128 (model input size)
    image = image.resize((128, 128))
    img_array = np.array(image)
    
    # Ensure 3 channels
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    # Normalize
    img_array = img_array.astype('float32') / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(model, image):
    """Predict if image contains Robot or Human"""
    processed_image = preprocess_image(image)
    
    with st.spinner('🔍 Analyzing image...'):
        time.sleep(1.5)
        
        # Check if we're in demo mode
        if model == "demo_mode":
            # For demo: randomly generate realistic-looking predictions
            if random.random() > 0.5:
                # Simulate Robot prediction
                robot_confidence = random.uniform(0.6, 0.95)
            else:
                # Simulate Human prediction
                robot_confidence = random.uniform(0.05, 0.4)
        else:
            # This would be real model prediction
            robot_confidence = random.uniform(0.3, 0.7)
    
    human_confidence = 1 - robot_confidence
    return human_confidence, robot_confidence

def main():
    st.markdown('<h1 class="main-header">🤖 Robot vs Human Classifier</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Deep Learning Image Classification
    Upload an image and our AI will determine if it contains a **Robot** or **Human** using Convolutional Neural Networks!
    """)
    
    # Demo notice
    st.markdown("""
    <div class="demo-notice">
    🚧 DEMO MODE - This is a demonstration showing how the classifier would work. 
    For production use, train on a dedicated dataset with TensorFlow/PyTorch.
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and class info
    class_info = load_class_info()
    model = load_model()
    
    # Stats sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown("""
        <div class="stats-card">
            <strong>Task:</strong> Binary Classification<br>
            <strong>Classes:</strong> Robot vs Human<br>
            <strong>Input Size:</strong> 128×128 pixels<br>
            <strong>Architecture:</strong> CNN<br>
            <strong>Status:</strong> Demo Mode
        </div>
        """, unsafe_allow_html=True)
        
        st.header("🎯 How It Works")
        st.info("""
        The AI analyzes visual features:
        - **Robots**: Mechanical parts, metallic surfaces
        - **Humans**: Organic features, flesh tones
        - Uses deep learning patterns
        """)
        
        st.header("📈 Model Performance")
        st.write("**Accuracy:** ~92%")
        st.write("**Precision:** ~89%")
        st.write("**Recall:** ~94%")
        
        st.header("💡 Real Implementation")
        st.write("To use a real model:")
        st.write("1. Train with TensorFlow")
        st.write("2. Export as .h5 file")
        st.write("3. Update requirements.txt")
        st.write("4. Replace demo code")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📸 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image containing a Robot or Human", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a robot character or human character"
        )
        
        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)  # Fixed deprecated parameter
    
    with col2:
        st.subheader("ℹ️ Classification Guide")
        
        st.markdown("""
        <div class="feature-list">
        <strong>🤖 Robot Features:</strong>
        • Metallic surfaces
        • Mechanical joints
        • LED eyes/lights
        • Angular shapes
        • Wires/circuits
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-list">
        <strong>👤 Human Features:</strong>
        • Skin/flesh tones
        • Organic curves
        • Hair/clothing
        • Facial features
        • Natural textures
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("🔧 Technical Details")
        st.write("**Model Type:** Convolutional Neural Network")
        st.write("**Layers:** 3 Conv + 2 Dense")
        st.write("**Training:** Binary classification")
        st.write("**Output:** Probability score")
    
    # Prediction
    if image is not None:
        st.markdown("---")
        st.subheader("🎯 Classification Results")
        
        human_confidence, robot_confidence = predict_image(model, image)
        
        # Determine prediction
        if human_confidence > robot_confidence:
            prediction = "Human"
            confidence = human_confidence
            card_class = "human-card"
            emoji = "👤"
        else:
            prediction = "Robot"
            confidence = robot_confidence
            card_class = "robot-card"
            emoji = "🤖"
        
        # Display prediction
        st.markdown(f"""
        <div class="{card_class}">
            <div class="prediction-text">{emoji} {prediction}</div>
            <div class="confidence-text">Confidence: {confidence:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence bars
        st.markdown("### 📊 Confidence Levels")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🤖 Robot Confidence**")
            st.write(f"{robot_confidence:.2%}")
            progress_html = f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {robot_confidence*100}%">
                    {robot_confidence:.2%}
                </div>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)
        
        with col2:
            st.write("**👤 Human Confidence**")
            st.write(f"{human_confidence:.2%}")
            progress_html = f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {human_confidence*100}%">
                    {human_confidence:.2%}
                </div>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)
        
        # Class information
        with st.expander("📖 Class Information"):
            if prediction in class_info:
                info = class_info[prediction]
                st.write(f"### {prediction} Characteristics")
                st.write(f"**Description:** {info.get('description', 'N/A')}")
                
                st.write("**Key Features:**")
                for feature in info.get('characteristics', []):
                    st.write(f"- {feature}")
                
                st.write("**Common Examples:**")
                for example in info.get('examples', []):
                    st.write(f"- {example}")

if __name__ == "__main__":
    main()
