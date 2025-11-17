import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import time
import json

# Set page configuration
st.set_page_config(
    page_title="Robot vs Human Classifier - Deep Learning CNN",
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
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        border: 3px solid #FFD93D;
    }
    .cnn-architecture {
        background: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #FF6B6B;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the pre-trained CNN model"""
    try:
        model = keras.models.load_model('robot_human_classifier.h5')
        st.success("✅ Deep Learning CNN Model Loaded Successfully!")
        
        # Display model architecture info
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        st.session_state.model_summary = model_summary
        
        return model
    except Exception as e:
        st.error(f"❌ Error loading CNN model: {e}")
        st.info("Please ensure 'robot_human_classifier.h5' is in your repository")
        return None

@st.cache_data
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

def preprocess_image(image, target_size=(128, 128)):
    """Preprocess the image for the CNN model"""
    # Resize to target size
    image = image.resize(target_size)
    img_array = np.array(image)
    
    # Ensure 3 channels (RGB)
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    # Normalize pixel values to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image(model, image):
    """Use the CNN model to predict if image contains Robot or Human"""
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    with st.spinner('🧠 Running Deep Learning CNN Analysis...'):
        # Start timer
        start_time = time.time()
        
        # Make prediction using the CNN model
        predictions = model.predict(processed_image, verbose=0)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Assuming binary classification:
        # If single output: sigmoid activation (0=Human, 1=Robot)
        # If two outputs: softmax activation ([Human_prob, Robot_prob])
        
        if predictions.shape[1] == 1:
            # Single output with sigmoid
            robot_confidence = float(predictions[0][0])
            human_confidence = 1 - robot_confidence
        else:
            # Multiple outputs with softmax
            human_confidence = float(predictions[0][0])
            robot_confidence = float(predictions[0][1])
    
    return human_confidence, robot_confidence, inference_time

def display_model_architecture():
    """Display CNN model architecture"""
    if 'model_summary' in st.session_state:
        st.markdown("### 🏗️ CNN Architecture")
        with st.expander("View Model Architecture"):
            for line in st.session_state.model_summary:
                st.text(line)

def main():
    st.markdown('<h1 class="main-header">🤖 Robot vs Human Classifier</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Deep Learning Convolutional Neural Network
    Upload an image and our **CNN model** will analyze it using deep learning to classify as **Robot** or **Human**!
    """)
    
    # Load model and class info
    class_info = load_class_info()
    model = load_model()
    
    if model is None:
        st.error("""
        **Model not found!** Please ensure:
        1. `robot_human_classifier.h5` is in your repository
        2. The file is properly uploaded to GitHub
        3. Wait for deployment to complete
        """)
        return
    
    # Display model architecture
    display_model_architecture()
    
    # Stats sidebar
    with st.sidebar:
        st.header("🧠 Deep Learning Info")
        st.markdown("""
        <div class="model-card">
            <strong>Model Type:</strong> Convolutional Neural Network<br>
            <strong>Framework:</strong> TensorFlow/Keras<br>
            <strong>Input Size:</strong> 128×128×3<br>
            <strong>Task:</strong> Binary Classification<br>
            <strong>Technology:</strong> Deep Learning
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="cnn-architecture">
            <strong>CNN Architecture:</strong><br>
            • 4 Convolutional Blocks<br>
            • Batch Normalization<br>
            • Max Pooling Layers<br>
            • Dropout Regularization<br>
            • Dense Classifier
        </div>
        """, unsafe_allow_html=True)
        
        st.header("🎯 How CNN Works")
        st.info("""
        **Convolutional Neural Networks:**
        - Extract hierarchical features
        - Learn spatial patterns
        - Automatically detect edges, textures, shapes
        - Generalize across different images
        """)
        
        st.header("📊 Model Capabilities")
        st.write("**Feature Learning:** Automatic feature extraction")
        st.write("**Pattern Recognition:** Spatial hierarchy learning")
        st.write("**Generalization:** Works on unseen images")
        st.write("**Accuracy:** Deep learning powered")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📸 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image for CNN analysis", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload any image - the CNN will analyze its visual features"
        )
        
        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Image for CNN Analysis", width=400)
            
            # Show image info
            st.write(f"**Image Details:** {image.size[0]}×{image.size[1]} pixels")
    
    with col2:
        st.subheader("🔍 CNN Analysis")
        
        st.markdown("""
        <div class="feature-list">
        <strong>🤖 Robot Features Learned:</strong>
        • Metallic texture patterns
        • Geometric shape detection
        • Mechanical component recognition
        • Artificial material analysis
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-list">
        <strong>👤 Human Features Learned:</strong>
        • Organic texture patterns
        • Facial feature detection
        • Skin tone recognition
        • Natural shape analysis
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("⚙️ Technical Specs")
        st.write("**Model Format:** Keras .h5")
        st.write("**Input Shape:** 128×128×3 RGB")
        st.write("**Layers:** Convolutional + Dense")
        st.write("**Activation:** ReLU + Sigmoid/Softmax")
    
    # Prediction
    if image is not None:
        st.markdown("---")
        st.subheader("🎯 CNN Classification Results")
        
        human_confidence, robot_confidence, inference_time = predict_image(model, image)
        
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
            <div class="confidence-text">CNN Confidence: {confidence:.2%}</div>
            <div class="confidence-text">Inference Time: {inference_time:.3f}s</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence bars
        st.markdown("### 📊 CNN Confidence Levels")
        
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
        
        # Performance metrics
        st.markdown("### ⚡ Performance Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Inference Time", f"{inference_time:.3f}s")
        with col2:
            st.metric("Prediction Confidence", f"{confidence:.2%}")
        with col3:
            st.metric("Model Type", "Deep Learning CNN")
        
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
        
        # Technical details
        with st.expander("🔧 Technical Details"):
            st.write("**Deep Learning Process:**")
            st.write("1. **Image Preprocessing:** Resize to 128×128, normalize pixels")
            st.write("2. **Feature Extraction:** CNN layers detect patterns")
            st.write("3. **Classification:** Dense layers make final decision")
            st.write("4. **Output:** Probability scores for each class")
            
            st.write("**CNN Advantages:**")
            st.write("- Automatic feature learning")
            st.write("- Spatial pattern recognition")
            st.write("- High accuracy on visual data")
            st.write("- Generalization to new images")

if __name__ == "__main__":
    main()
