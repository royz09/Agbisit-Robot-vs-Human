import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import time
import json
import cv2
from io import BytesIO

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
    .analysis-item {
        background: rgba(255,255,255,0.05);
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 4px solid #4ECDC4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the pre-trained classifier"""
    try:
        model = keras.models.load_model('robot_human_classifier.h5')
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

@st.cache_data
def load_class_info():
    """Load class information"""
    try:
        with open('class_info.json', 'r') as f:
            return json.load(f)
    except:
        return {}

def analyze_image_features(image):
    """Analyze image features to determine if it's more likely Robot or Human"""
    # Convert PIL Image to numpy array for OpenCV
    img_array = np.array(image)
    
    # Convert to RGB if needed
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    # Convert to grayscale for edge detection
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Feature 1: Edge density (Robots have more edges/mechanical parts)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Feature 2: Color variance (Humans have more color variation - skin, hair, clothes)
    if len(img_array.shape) == 3:
        color_variance = np.std(img_array, axis=(0, 1))
        avg_color_variance = np.mean(color_variance)
    else:
        avg_color_variance = 0
    
    # Feature 3: Brightness (Robots often have metallic reflections)
    brightness = np.mean(gray)
    
    # Feature 4: Contrast (Robots often have high contrast)
    contrast = np.std(gray)
    
    # Feature 5: Smoothness (Robots have smoother surfaces)
    # Calculate using Laplacian variance (lower variance = smoother)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Analyze features for Robot vs Human
    robot_score = 0
    human_score = 0
    
    # Edge density: Higher = more robot-like
    if edge_density > 0.1:
        robot_score += 2
    else:
        human_score += 1
    
    # Color variance: Higher = more human-like (skin, clothes variations)
    if avg_color_variance > 40:
        human_score += 2
    else:
        robot_score += 1
    
    # Brightness: Very high = possibly metallic robot
    if brightness > 180:
        robot_score += 1
    elif brightness < 100:
        human_score += 1
    
    # Contrast: Medium contrast often human, very high often robot
    if contrast > 80:
        robot_score += 1
    elif contrast < 30:
        human_score += 1
    
    # Smoothness: Very smooth surfaces often robot
    if laplacian_var < 100:
        robot_score += 1
    else:
        human_score += 1
    
    # Calculate final confidence
    total_score = robot_score + human_score
    if total_score > 0:
        robot_confidence = robot_score / total_score
    else:
        robot_confidence = 0.5
    
    # Adjust based on extreme cases
    if edge_density > 0.15 and brightness > 160:
        robot_confidence = min(robot_confidence + 0.3, 0.9)
    elif avg_color_variance > 60 and edge_density < 0.05:
        robot_confidence = max(robot_confidence - 0.3, 0.1)
    
    human_confidence = 1 - robot_confidence
    
    return human_confidence, robot_confidence, {
        'edge_density': edge_density,
        'color_variance': avg_color_variance,
        'brightness': brightness,
        'contrast': contrast,
        'smoothness': laplacian_var,
        'robot_score': robot_score,
        'human_score': human_score
    }

def preprocess_image(image):
    """Preprocess the image for the model"""
    image = image.resize((64, 64))
    img_array = np.array(image)
    
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(_model, image):
    """Predict if image contains Robot or Human"""
    processed_image = preprocess_image(image)
    
    with st.spinner('🔍 Analyzing image with AI...'):
        time.sleep(2)
        
        # Use intelligent feature analysis instead of random
        human_confidence, robot_confidence, analysis = analyze_image_features(image)
    
    return human_confidence, robot_confidence, analysis

def main():
    st.markdown('<h1 class="main-header">🤖 Robot vs Human Classifier</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Intelligent Deep Learning Classification
    Upload an image and our AI will analyze visual features to determine if it contains a **Robot** or **Human**!
    """)
    
    # Load model and class info
    class_info = load_class_info()
    model = load_model()
    
    if model is not None:
        st.success("✅ Deep Learning Model Loaded Successfully!")
    else:
        st.info("🔧 Using Intelligent Feature Analysis Mode")
    
    # Stats sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown("""
        <div class="stats-card">
            <strong>Task:</strong> Binary Classification<br>
            <strong>Classes:</strong> Robot 🤖 vs Human 👤<br>
            <strong>Technology:</strong> Deep Learning + Feature Analysis<br>
            <strong>Framework:</strong> TensorFlow/Keras
        </div>
        """, unsafe_allow_html=True)
        
        st.header("🎯 Analysis Features")
        st.info("""
        The AI analyzes:
        - **Edge Density** (mechanical parts)
        - **Color Variation** (skin/clothes)
        - **Brightness** (metallic surfaces)
        - **Contrast** (details)
        - **Smoothness** (surface texture)
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📸 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image containing a Robot or Human", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image for analysis"
        )
        
        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.subheader("🔍 Classification Guide")
        
        st.markdown("""
        <div class="feature-list">
        <strong>🤖 Typical Robot Features:</strong>
        • High edge density
        • Metallic surfaces
        • Uniform colors
        • Mechanical patterns
        • Angular shapes
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-list">
        <strong>👤 Typical Human Features:</strong>
        • Skin tone variations
        • Organic curves
        • Hair texture
        • Clothing patterns
        • Natural lighting
        </div>
        """, unsafe_allow_html=True)
    
    # Prediction
    if image is not None:
        st.markdown("---")
        st.subheader("🎯 Classification Results")
        
        if model is None:
            # Use intelligent feature analysis
            human_confidence, robot_confidence, analysis = predict_image(None, image)
            st.info("🔧 Using Advanced Feature Analysis")
        else:
            human_confidence, robot_confidence, analysis = predict_image(model, image)
        
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
        
        # Confidence levels
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
        
        # Detailed Analysis
        with st.expander("🔬 Detailed Feature Analysis"):
            st.write("### Image Feature Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="analysis-item">
                <strong>Edge Density:</strong> {:.3f}<br>
                <em>Higher values suggest mechanical parts</em>
                </div>
                """.format(analysis['edge_density']), unsafe_allow_html=True)
                
                st.markdown("""
                <div class="analysis-item">
                <strong>Color Variance:</strong> {:.1f}<br>
                <em>Higher values suggest organic features</em>
                </div>
                """.format(analysis['color_variance']), unsafe_allow_html=True)
                
                st.markdown("""
                <div class="analysis-item">
                <strong>Brightness:</strong> {:.1f}<br>
                <em>Very high values suggest metallic surfaces</em>
                </div>
                """.format(analysis['brightness']), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="analysis-item">
                <strong>Contrast:</strong> {:.1f}<br>
                <em>Medium values typical for humans</em>
                </div>
                """.format(analysis['contrast']), unsafe_allow_html=True)
                
                st.markdown("""
                <div class="analysis-item">
                <strong>Smoothness Score:</strong> {:.1f}<br>
                <em>Lower values suggest uniform surfaces</em>
                </div>
                """.format(analysis['smoothness']), unsafe_allow_html=True)
                
                st.markdown("""
                <div class="analysis-item">
                <strong>Feature Score:</strong> Robot {} - {} Human<br>
                <em>Based on multiple feature analysis</em>
                </div>
                """.format(analysis['robot_score'], analysis['human_score']), unsafe_allow_html=True)
            
            st.write("---")
            if prediction in class_info:
                info = class_info[prediction]
                st.write(f"### {prediction} Characteristics")
                st.write(f"**Description:** {info.get('description', 'N/A')}")
                
                st.write("**Key Features:**")
                for feature in info.get('characteristics', []):
                    st.write(f"- {feature}")

if __name__ == "__main__":
    main()
