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
    .robot-indicator {
        border-left: 4px solid #FF6B6B !important;
    }
    .human-indicator {
        border-left: 4px solid #4ECDC4 !important;
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
        return None

@st.cache_data
def load_class_info():
    """Load class information"""
    try:
        with open('class_info.json', 'r') as f:
            return json.load(f)
    except:
        return {}

def detect_faces(image_array):
    """Detect if there are human faces in the image"""
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Load face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        return len(faces) > 0, len(faces)
    except:
        return False, 0

def detect_skin_regions(image_array):
    """Detect skin-colored regions (common in humans)"""
    try:
        # Convert to HSV color space
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        
        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create skin mask
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Calculate percentage of skin pixels
        skin_ratio = np.sum(skin_mask > 0) / (skin_mask.shape[0] * skin_mask.shape[1])
        
        return skin_ratio > 0.05, skin_ratio
    except:
        return False, 0

def detect_metallic_regions(image_array):
    """Detect metallic/robotic looking regions"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Metallic regions often have high brightness and specific texture
        # Look for very bright regions (potential metallic reflections)
        bright_threshold = 200
        bright_pixels = np.sum(gray > bright_threshold)
        bright_ratio = bright_pixels / (gray.shape[0] * gray.shape[1])
        
        # Look for uniform regions (common in robots)
        uniform_threshold = 20
        local_std = cv2.blur(gray, (5, 5))
        local_std = cv2.absdiff(gray, local_std)
        uniform_pixels = np.sum(local_std < uniform_threshold)
        uniform_ratio = uniform_pixels / (gray.shape[0] * gray.shape[1])
        
        return bright_ratio > 0.1 or uniform_ratio > 0.3, bright_ratio, uniform_ratio
    except:
        return False, 0, 0

def analyze_geometric_shapes(image_array):
    """Analyze geometric shapes (robots often have more geometric features)"""
    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines (robots often have straight lines)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        line_count = 0 if lines is None else len(lines)
        
        # Detect circles (less common in robots)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=100)
        circle_count = 0 if circles is None else len(circles[0])
        
        # Geometric ratio (more lines = more robot-like)
        total_pixels = edges.shape[0] * edges.shape[1]
        line_density = line_count / total_pixels * 10000  # Normalize
        
        return line_density, line_count, circle_count
    except:
        return 0, 0, 0

def analyze_image_features(image):
    """Advanced analysis of image features for Robot vs Human classification"""
    img_array = np.array(image)
    
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    # Feature 1: Face detection (strong human indicator)
    has_faces, face_count = detect_faces(img_array)
    
    # Feature 2: Skin detection (human indicator)
    has_skin, skin_ratio = detect_skin_regions(img_array)
    
    # Feature 3: Metallic regions (robot indicator)
    has_metallic, bright_ratio, uniform_ratio = detect_metallic_regions(img_array)
    
    # Feature 4: Geometric shapes analysis
    line_density, line_count, circle_count = analyze_geometric_shapes(img_array)
    
    # Feature 5: Color analysis
    if len(img_array.shape) == 3:
        color_std = np.std(img_array, axis=(0, 1))
        avg_color_std = np.mean(color_std)
    else:
        avg_color_std = 0
    
    # Feature 6: Edge density
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Scoring system
    robot_score = 0
    human_score = 0
    
    # Strong indicators
    if has_faces:
        human_score += 3
        st.info(f"👤 Detected {face_count} face(s) - Strong human indicator")
    
    if has_skin and skin_ratio > 0.1:
        human_score += 2
        st.info(f"🎨 Detected skin tones ({skin_ratio:.1%}) - Human indicator")
    
    if has_metallic:
        robot_score += 3
        st.info(f"🔧 Detected metallic features - Strong robot indicator")
    
    # Moderate indicators
    if line_density > 2.0:  # High line density = mechanical
        robot_score += 2
        st.info(f"📐 High geometric line density ({line_density:.1f}) - Robot indicator")
    
    if edge_density > 0.15:  # Very high edge density = mechanical
        robot_score += 1
    elif edge_density < 0.05:  # Very low = likely organic
        human_score += 1
    
    if avg_color_std < 30:  # Low color variation = uniform surfaces (robot)
        robot_score += 1
    elif avg_color_std > 60:  # High color variation = organic (human)
        human_score += 1
    
    if uniform_ratio > 0.4:  # Very uniform regions = robot
        robot_score += 2
    
    if bright_ratio > 0.15:  # Very bright regions = metallic reflections
        robot_score += 1
    
    # Calculate final confidence
    total_score = max(robot_score + human_score, 1)  # Avoid division by zero
    
    # Base robot confidence on robot score proportion
    base_robot_confidence = robot_score / total_score
    
    # Apply strong feature boosts
    if has_faces and human_score > robot_score:
        base_robot_confidence = max(base_robot_confidence - 0.3, 0.1)
    
    if has_metallic and robot_score > human_score:
        base_robot_confidence = min(base_robot_confidence + 0.3, 0.9)
    
    # Ensure reasonable bounds
    robot_confidence = max(0.1, min(0.9, base_robot_confidence))
    human_confidence = 1 - robot_confidence
    
    return human_confidence, robot_confidence, {
        'faces_detected': face_count,
        'skin_ratio': skin_ratio,
        'metallic_detected': has_metallic,
        'line_density': line_density,
        'edge_density': edge_density,
        'color_variance': avg_color_std,
        'bright_ratio': bright_ratio,
        'uniform_ratio': uniform_ratio,
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
    
    with st.spinner('🔍 Advanced AI analysis in progress...'):
        time.sleep(2)
        
        # Use advanced feature analysis
        human_confidence, robot_confidence, analysis = analyze_image_features(image)
    
    return human_confidence, robot_confidence, analysis

def main():
    st.markdown('<h1 class="main-header">🤖 Robot vs Human Classifier</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Advanced AI Classification
    Upload an image and our AI will perform **multi-feature analysis** to determine if it contains a **Robot** or **Human**!
    """)
    
    # Load model and class info
    class_info = load_class_info()
    model = load_model()
    
    if model is not None:
        st.success("✅ Deep Learning Model Loaded Successfully!")
    else:
        st.info("🔧 Using Advanced Multi-Feature Analysis")
    
    # Stats sidebar
    with st.sidebar:
        st.header("📊 Analysis Features")
        st.markdown("""
        <div class="stats-card">
        <strong>🤖 Robot Detection:</strong>
        • Metallic regions
        • Geometric shapes
        • Uniform surfaces
        • High edge density
        • Bright reflections
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="stats-card">
        <strong>👤 Human Detection:</strong>
        • Face detection
        • Skin tones
        • Organic shapes
        • Color variation
        • Natural textures
        </div>
        """, unsafe_allow_html=True)
    
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
        st.subheader("🎯 Test Images")
        st.info("""
        **For best results, try:**
        
        **🤖 Robot Images:**
        • Mechanical robots
        • Android characters
        • Metal surfaces
        • Geometric shapes
        
        **👤 Human Images:**
        • People's faces
        • Skin showing
        • Natural scenes
        • Organic shapes
        """)
    
    # Prediction
    if image is not None:
        st.markdown("---")
        st.subheader("🎯 Classification Results")
        
        # Clear previous analysis messages
        if 'analysis_messages' in st.session_state:
            del st.session_state['analysis_messages']
        
        if model is None:
            human_confidence, robot_confidence, analysis = predict_image(None, image)
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
            st.write("### Feature Analysis Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Robot indicators
                st.write("**🤖 Robot Indicators:**")
                if analysis['metallic_detected']:
                    st.markdown("""
                    <div class="analysis-item robot-indicator">
                    <strong>Metallic Features:</strong> Detected<br>
                    <em>Strong robot indicator</em>
                    </div>
                    """, unsafe_allow_html=True)
                
                if analysis['line_density'] > 1.5:
                    st.markdown("""
                    <div class="analysis-item robot-indicator">
                    <strong>Geometric Lines:</strong> {:.1f} density<br>
                    <em>Mechanical structure detected</em>
                    </div>
                    """.format(analysis['line_density']), unsafe_allow_html=True)
                
                if analysis['edge_density'] > 0.12:
                    st.markdown("""
                    <div class="analysis-item robot-indicator">
                    <strong>Edge Density:</strong> {:.1%}<br>
                    <em>High detail suggests mechanical parts</em>
                    </div>
                    """.format(analysis['edge_density']), unsafe_allow_html=True)
                
                if analysis['uniform_ratio'] > 0.3:
                    st.markdown("""
                    <div class="analysis-item robot-indicator">
                    <strong>Uniform Regions:</strong> {:.1%}<br>
                    <em>Consistent surfaces suggest manufactured object</em>
                    </div>
                    """.format(analysis['uniform_ratio']), unsafe_allow_html=True)
            
            with col2:
                # Human indicators
                st.write("**👤 Human Indicators:**")
                if analysis['faces_detected'] > 0:
                    st.markdown("""
                    <div class="analysis-item human-indicator">
                    <strong>Faces Detected:</strong> {}<br>
                    <em>Strong human indicator</em>
                    </div>
                    """.format(analysis['faces_detected']), unsafe_allow_html=True)
                
                if analysis['skin_ratio'] > 0.05:
                    st.markdown("""
                    <div class="analysis-item human-indicator">
                    <strong>Skin Regions:</strong> {:.1%}<br>
                    <em>Skin tones detected</em>
                    </div>
                    """.format(analysis['skin_ratio']), unsafe_allow_html=True)
                
                if analysis['color_variance'] > 50:
                    st.markdown("""
                    <div class="analysis-item human-indicator">
                    <strong>Color Variation:</strong> {:.1f}<br>
                    <em>High variation suggests organic features</em>
                    </div>
                    """.format(analysis['color_variance']), unsafe_allow_html=True)
            
            st.write("---")
            st.write(f"**Final Score:** Robot {analysis['robot_score']} - {analysis['human_score']} Human")
            
            if prediction in class_info:
                info = class_info[prediction]
                st.write(f"### {prediction} Characteristics")
                st.write(f"**Description:** {info.get('description', 'N/A')}")

if __name__ == "__main__":
    main()
