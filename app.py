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
    }
    .robot-indicator {
        border-left: 4px solid #FF6B6B !important;
    }
    .human-indicator {
        border-left: 4px solid #4ECDC4 !important;
    }
    .neutral-indicator {
        border-left: 4px solid #FFD93D !important;
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
    """Detect human faces with confidence"""
    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return len(faces) > 0, len(faces)
    except:
        return False, 0

def analyze_color_patterns(image_array):
    """Analyze color patterns for organic vs mechanical appearance"""
    try:
        # Convert to different color spaces
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        
        # Organic features (humans)
        # Skin tone detection in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
        
        # Natural color variance (humans have more color variation)
        color_std = np.std(image_array, axis=(0, 1))
        color_variance = np.mean(color_std)
        
        # Mechanical features (robots)
        # Metallic/silver detection
        lower_metal = np.array([0, 0, 100], dtype=np.uint8)  # Bright, low saturation
        upper_metal = np.array([180, 50, 255], dtype=np.uint8)
        metal_mask = cv2.inRange(hsv, lower_metal, upper_metal)
        metal_ratio = np.sum(metal_mask > 0) / metal_mask.size
        
        # Uniform color regions (common in robots)
        from scipy import ndimage
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        local_std = ndimage.generic_filter(gray, np.std, size=5)
        uniform_ratio = np.sum(local_std < 15) / local_std.size
        
        return {
            'skin_ratio': skin_ratio,
            'color_variance': color_variance,
            'metal_ratio': metal_ratio,
            'uniform_ratio': uniform_ratio
        }
    except:
        return {'skin_ratio': 0, 'color_variance': 0, 'metal_ratio': 0, 'uniform_ratio': 0}

def analyze_texture_and_edges(image_array):
    """Analyze texture and edge patterns"""
    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Edge analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Texture analysis using Local Binary Patterns
        from skimage import feature
        lbp = feature.local_binary_pattern(gray, 24, 3, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 25))
        lbp_hist = lbp_hist / lbp_hist.sum()  # Normalize
        texture_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-8))  # Texture complexity
        
        # Geometric shape detection
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        geometric_score = 0
        if contours:
            for contour in contours:
                if len(contour) >= 5:
                    # Fit ellipse to check for organic shapes
                    ellipse = cv2.fitEllipse(contour)
                    # Circular/elliptical shapes are more organic
                    if ellipse[1][0] > 0 and ellipse[1][1] > 0:
                        aspect_ratio = max(ellipse[1]) / min(ellipse[1])
                        if 0.8 < aspect_ratio < 1.2:  # Near circular
                            geometric_score -= 1  # More organic
                        else:
                            geometric_score += 0.5  # More mechanical
        
        return {
            'edge_density': edge_density,
            'texture_entropy': texture_entropy,
            'geometric_score': geometric_score
        }
    except:
        return {'edge_density': 0, 'texture_entropy': 0, 'geometric_score': 0}

def analyze_brightness_and_contrast(image_array):
    """Analyze brightness and contrast patterns"""
    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Brightness analysis
        brightness = np.mean(gray)
        
        # Contrast analysis
        contrast = np.std(gray)
        
        # Bright pixel ratio (metallic reflections)
        bright_ratio = np.sum(gray > 200) / gray.size
        
        # Dark pixel ratio (shadows, common in organic scenes)
        dark_ratio = np.sum(gray < 50) / gray.size
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'bright_ratio': bright_ratio,
            'dark_ratio': dark_ratio
        }
    except:
        return {'brightness': 0, 'contrast': 0, 'bright_ratio': 0, 'dark_ratio': 0}

def balanced_classification(image):
    """Perform balanced analysis considering both robot and human features"""
    img_array = np.array(image)
    
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    # Reset analysis messages
    analysis_messages = []
    
    # 1. Face detection (strong human indicator)
    has_faces, face_count = detect_faces(img_array)
    if has_faces:
        analysis_messages.append(f"👤 Detected {face_count} face(s) - Strong human indicator")
    
    # 2. Color pattern analysis
    color_analysis = analyze_color_patterns(img_array)
    
    # 3. Texture and edge analysis
    texture_analysis = analyze_texture_and_edges(img_array)
    
    # 4. Brightness and contrast analysis
    brightness_analysis = analyze_brightness_and_contrast(img_array)
    
    # Scoring system - BALANCED APPROACH
    human_score = 0
    robot_score = 0
    
    # Strong indicators
    if has_faces:
        human_score += 4  # Very strong human indicator
    
    # Color-based scoring
    if color_analysis['skin_ratio'] > 0.05:
        human_score += 3
        analysis_messages.append(f"🎨 Skin tones detected ({color_analysis['skin_ratio']:.1%}) - Human indicator")
    
    if color_analysis['metal_ratio'] > 0.08:
        robot_score += 3
        analysis_messages.append(f"🔩 Metallic surfaces detected ({color_analysis['metal_ratio']:.1%}) - Robot indicator")
    
    if color_analysis['color_variance'] > 60:
        human_score += 2
        analysis_messages.append("🌈 High color variation - Organic features")
    elif color_analysis['color_variance'] < 25:
        robot_score += 2
        analysis_messages.append("⚫ Low color variation - Uniform surfaces")
    
    # Texture-based scoring
    if texture_analysis['edge_density'] > 0.12:
        robot_score += 2
        analysis_messages.append(f"📐 High edge density ({texture_analysis['edge_density']:.1%}) - Mechanical details")
    elif texture_analysis['edge_density'] < 0.04:
        human_score += 1
        analysis_messages.append("🟢 Low edge density - Smooth organic surfaces")
    
    if texture_analysis['texture_entropy'] > 2.5:
        human_score += 2
        analysis_messages.append("🌀 Complex texture patterns - Organic material")
    elif texture_analysis['texture_entropy'] < 1.5:
        robot_score += 1
        analysis_messages.append("🔲 Simple texture patterns - Manufactured surface")
    
    # Brightness-based scoring
    if brightness_analysis['bright_ratio'] > 0.1:
        robot_score += 2
        analysis_messages.append(f"💡 High brightness areas ({brightness_analysis['bright_ratio']:.1%}) - Metallic reflections")
    
    if brightness_analysis['dark_ratio'] > 0.15:
        human_score += 1
        analysis_messages.append("🌑 Shadow areas detected - Natural lighting")
    
    # Geometric scoring
    if texture_analysis['geometric_score'] > 2:
        robot_score += 2
        analysis_messages.append("📏 Geometric shapes detected - Mechanical design")
    elif texture_analysis['geometric_score'] < -1:
        human_score += 1
        analysis_messages.append("🔵 Organic shapes detected - Natural forms")
    
    # Calculate final confidence with smoothing
    total_score = human_score + robot_score
    if total_score == 0:
        # If no strong indicators, default to uncertain
        human_confidence = 0.5
        robot_confidence = 0.5
    else:
        base_human_confidence = human_score / total_score
        base_robot_confidence = robot_score / total_score
        
        # Apply moderate adjustments based on strong features
        if has_faces and human_score > robot_score:
            base_human_confidence = min(base_human_confidence + 0.2, 0.9)
        elif color_analysis['metal_ratio'] > 0.1 and robot_score > human_score:
            base_robot_confidence = min(base_robot_confidence + 0.2, 0.9)
        
        human_confidence = max(0.1, min(0.9, base_human_confidence))
        robot_confidence = 1 - human_confidence
    
    # Compile analysis results
    analysis_results = {
        'faces_detected': face_count,
        'skin_ratio': color_analysis['skin_ratio'],
        'metal_ratio': color_analysis['metal_ratio'],
        'color_variance': color_analysis['color_variance'],
        'edge_density': texture_analysis['edge_density'],
        'texture_entropy': texture_analysis['texture_entropy'],
        'bright_ratio': brightness_analysis['bright_ratio'],
        'human_score': human_score,
        'robot_score': robot_score,
        'messages': analysis_messages
    }
    
    return human_confidence, robot_confidence, analysis_results

def main():
    st.markdown('<h1 class="main-header">🤖 Robot vs Human Classifier</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Balanced AI Classification
    Upload an image for **multi-feature analysis** that fairly evaluates both Robot and Human characteristics!
    """)
    
    # Load model and class info
    class_info = load_class_info()
    model = load_model()
    
    if model is not None:
        st.success("✅ Deep Learning Model Loaded Successfully!")
    else:
        st.info("🔧 Using Advanced Balanced Feature Analysis")
    
    # Stats sidebar
    with st.sidebar:
        st.header("🎯 How It Works")
        st.markdown("""
        <div class="stats-card">
        The AI analyzes multiple features:
        
        **🤖 Robot Indicators:**
        • Metallic surfaces
        • Geometric shapes  
        • Uniform colors
        • High edge density
        • Bright reflections
        
        **👤 Human Indicators:**
        • Face detection
        • Skin tones
        • Color variation
        • Complex textures
        • Organic shapes
        </div>
        """, unsafe_allow_html=True)
        
        st.header("💡 Tips")
        st.info("""
        For best results:
        - Use clear, well-lit images
        - Center the main subject
        - Avoid blurry images
        - Good contrast helps
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📸 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of a robot or human"
        )
        
        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.subheader("🔍 Test Suggestions")
        st.info("""
        **Try these for testing:**
        
        **Clear Robots:**
        • Mechanical robots
        • Metal surfaces
        • Android characters
        
        **Clear Humans:**
        • Portrait photos
        • People with visible skin
        • Natural scenes with people
        """)
    
    # Prediction
    if image is not None:
        st.markdown("---")
        st.subheader("🎯 Classification Results")
        
        with st.spinner('🔍 Performing balanced analysis...'):
            time.sleep(2)
            human_confidence, robot_confidence, analysis = balanced_classification(image)
        
        # Show analysis messages
        if analysis['messages']:
            st.write("**Feature Analysis:**")
            for msg in analysis['messages']:
                st.write(f"• {msg}")
        
        # Determine prediction
        confidence_threshold = 0.1  # Small buffer to avoid 50/50 ties
        if human_confidence > robot_confidence + confidence_threshold:
            prediction = "Human"
            confidence = human_confidence
            card_class = "human-card"
            emoji = "👤"
        elif robot_confidence > human_confidence + confidence_threshold:
            prediction = "Robot"
            confidence = robot_confidence
            card_class = "robot-card"
            emoji = "🤖"
        else:
            prediction = "Uncertain"
            confidence = max(human_confidence, robot_confidence)
            card_class = "stats-card"
            emoji = "❓"
        
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
        with st.expander("🔬 Technical Analysis Details"):
            st.write("### Feature Measurements")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="analysis-item {'human-indicator' if analysis['faces_detected'] > 0 else 'neutral-indicator'}">
                <strong>Faces Detected:</strong> {analysis['faces_detected']}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="analysis-item {'human-indicator' if analysis['skin_ratio'] > 0.05 else 'neutral-indicator'}">
                <strong>Skin Ratio:</strong> {analysis['skin_ratio']:.3f}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="analysis-item {'robot-indicator' if analysis['metal_ratio'] > 0.08 else 'neutral-indicator'}">
                <strong>Metal Ratio:</strong> {analysis['metal_ratio']:.3f}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="analysis-item {'human-indicator' if analysis['color_variance'] > 60 else 'robot-indicator' if analysis['color_variance'] < 25 else 'neutral-indicator'}">
                <strong>Color Variance:</strong> {analysis['color_variance']:.1f}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="analysis-item {'robot-indicator' if analysis['edge_density'] > 0.12 else 'human-indicator' if analysis['edge_density'] < 0.04 else 'neutral-indicator'}">
                <strong>Edge Density:</strong> {analysis['edge_density']:.3f}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="analysis-item {'robot-indicator' if analysis['bright_ratio'] > 0.1 else 'neutral-indicator'}">
                <strong>Bright Areas:</strong> {analysis['bright_ratio']:.3f}
                </div>
                """, unsafe_allow_html=True)
            
            st.write("---")
            st.write(f"**Final Score:** Human {analysis['human_score']} - {analysis['robot_score']} Robot")
            
            if prediction != "Uncertain" and prediction in class_info:
                info = class_info[prediction]
                st.write(f"### {prediction} Characteristics")
                st.write(f"**Description:** {info.get('description', 'N/A')}")

if __name__ == "__main__":
    main()
