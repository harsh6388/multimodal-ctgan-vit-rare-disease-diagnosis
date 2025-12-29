# ===============================================================
# 🫁 Lung Cancer Risk Prediction App (Tabular + CT Scan Fusion)
# Author: Harsh Verma
# Enhanced Version with Beautiful UI
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import joblib
import sqlite3
import hashlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler





# 🗂️ Database Setup
# ===============================================================
conn = sqlite3.connect('patients.db', check_same_thread=False)
c = conn.cursor()

# Create users table
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE,
    password TEXT
)
''')
conn.commit()

# ===============================================================
# 🔒 Helper Functions
# ===============================================================
def make_hash(password):
    return hashlib.sha256(str(password).encode()).hexdigest()

def add_user(name, email, password):
    c.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)', 
              (name, email, make_hash(password)))
    conn.commit()

def login_user(email, password):
    c.execute('SELECT * FROM users WHERE email = ? AND password = ?', 
              (email, make_hash(password)))
    return c.fetchone()


# ===============================================================
# 🎨 Custom CSS for Beautiful UI
# ===============================================================
st.markdown("""
<style>
    /* Main background with gradient */
    .stApp {
        background: linear-gradient(135deg, #E8F5E9 0%, #E3F2FD 50%, #F3E5F5 100%);
    }
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(90deg, #66BB6A 0%, #42A5F5 50%, #AB47BC 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Section containers */
    .section-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border-left: 5px solid #66BB6A;
    }
    
    /* Upload section styling */
    .upload-section {
        background: linear-gradient(135deg, #E1F5FE 0%, #F3E5F5 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #42A5F5;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Result cards */
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border-left: 4px solid #66BB6A;
    }
    
    /* Success message styling */
    .success-box {
        background: linear-gradient(135deg, #C8E6C9 0%, #A5D6A7 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    
    /* Warning message styling */
    .warning-box {
        background: linear-gradient(135deg, #FFE082 0%, #FFD54F 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #FFA726;
        margin: 1rem 0;
    }
    
    /* Error message styling */
    .error-box {
        background: linear-gradient(135deg, #FFCDD2 0%, #EF9A9A 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #EF5350;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #66BB6A 0%, #42A5F5 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        border: none;
        box-shadow: 0 4px 12px rgba(66, 165, 245, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(66, 165, 245, 0.6);
    }
    
    /* Info boxes */
    .info-card {
        background: linear-gradient(135deg, #E1F5FE 0%, #B3E5FC 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #42A5F5;
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #66BB6A, transparent);
        margin: 2rem 0;
    }
    
    /* Input labels */
    .stSelectbox label, .stSlider label {
        color: #2E7D32;
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* DataFrame styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Footer styling */
    .footer {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 2rem;
        border-top: 3px solid #66BB6A;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ===============================================================
# 🧠 Load trained models
# ===============================================================
try:
    rf_model = joblib.load("lung_cancer_model.pkl")  # RandomForest
    scaler = joblib.load("scaler.pkl")               # StandardScaler
except:
    st.warning("⚠️ No pre-trained tabular model found. Using placeholder.")
    rf_model = RandomForestClassifier(random_state=42)
    scaler = StandardScaler()

# Load Vision Transformer model
@st.cache_resource
def load_vit_model():
    vit = models.vit_b_16(weights=None)
    vit.heads.head = nn.Linear(vit.heads.head.in_features, 2)
    vit.load_state_dict(torch.load("vit_lung_cancer.pth", map_location="cpu"))
    vit.eval()
    weights = models.ViT_B_16_Weights.DEFAULT
    transform = weights.transforms()
    return vit, transform

vit, vit_transform = load_vit_model()

# ===============================================================
# 👥 Authentication (Signup/Login)
# ===============================================================
st.sidebar.title("🔐 Patient Login System")

menu = ["Login", "Sign Up"]
choice = st.sidebar.radio("Choose Action", menu)

if choice == "Sign Up":
    st.sidebar.subheader("Create a new account")
    name = st.sidebar.text_input("Full Name")
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Sign Up"):
        try:
            add_user(name, email, password)
            st.sidebar.success("✅ Account created successfully! You can now log in.")
        except:
            st.sidebar.error("⚠️ Email already exists or invalid entry.")

elif choice == "Login":
    st.sidebar.subheader("Login to your account")
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        user = login_user(email, password)
        if user:
            st.session_state["logged_in"] = True
            st.session_state["user"] = user
            st.sidebar.success(f"Welcome, {user[1]} 👋")
        else:
            st.sidebar.error("❌ Invalid email or password.")





# ===============================================================
# 🎨 Streamlit UI
# ===============================================================
st.set_page_config(
    page_title="Lung Cancer Risk Predictor 🫁", 
    page_icon="🫁", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===============================================================
# 🔐 Require Login to Access App
# ===============================================================
if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    st.warning("⚠️ Please log in from the sidebar to access the Lung Cancer Predictor.")
    st.stop()




# Custom header
st.markdown("""
<div class="main-header">
    <h1>🫁 ViT-CTGAN Assisted Lung Diagnostic Web App</h1>
    <p>Advanced AI-Powered Health Assessment Tool</p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem;">Upload your CT scan and health information for personalized risk analysis</p>
</div>
""", unsafe_allow_html=True)

# ===============================================================
# 📸 Upload CT Scan
# ===============================================================
st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.markdown("### 📷 CT Scan Image Upload")
st.markdown("Upload a chest CT scan image for AI-powered image analysis")

uploaded_image = st.file_uploader(
    "Choose your lung CT scan (JPG or PNG)", 
    type=["jpg", "jpeg", "png"],
    help="Upload a clear CT scan image of the lungs"
)

image_prob = None
if uploaded_image is not None:
    img = Image.open(uploaded_image).convert("RGB")
    
    col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
    with col_img2:
        st.image(img, caption="📊 Uploaded CT Scan", use_container_width=True)
    
    with st.spinner("🔍 Analyzing CT scan with AI..."):
        with torch.no_grad():
            x = vit_transform(img).unsqueeze(0)
            probs = torch.softmax(vit(x), dim=1).numpy()[0]
            cancer_prob = probs[0]
            image_prob = float(cancer_prob)
    
    st.markdown(f"""
    <div class="info-card">
        <h4 style="margin: 0; color: #1976D2;">🧠 AI Image Analysis Complete</h4>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            CT Scan Risk Probability: <strong>{image_prob:.1%}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ===============================================================
# 🧾 Patient Info (Tabular Data)
# ===============================================================
st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.markdown("### 📋 Patient Health Information")
st.markdown("Please provide accurate information for better risk assessment")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 👤 Demographics & Lifestyle")
    AGE = st.slider("Age", 18, 90, 45, help="Patient's current age")
    SMOKING = st.selectbox("Smoking", ["No", "Yes"], help="Current or past smoking habit")
    ALCOHOL_CONSUMING = st.selectbox("Alcohol Consumption", ["No", "Yes"], help="Regular alcohol consumption")
    YELLOW_FINGERS = st.selectbox("Yellow Fingers", ["No", "Yes"], help="Yellowing of fingers (often from smoking)")
    
    st.markdown("#### 🧠 Psychological Factors")
    ANXIETY = st.selectbox("Anxiety", ["No", "Yes"], help="Experiencing anxiety or stress")
    PEER_PRESSURE = st.selectbox("Peer Pressure", ["No", "Yes"], help="Influenced by peer pressure")

with col2:
    st.markdown("#### 🩺 Medical History")
    CHRONIC_DISEASE = st.selectbox("Chronic Disease", ["No", "Yes"], help="Any chronic medical conditions")
    ALLERGY = st.selectbox("Allergy", ["No", "Yes"], help="Known allergies")
    FATIGUE = st.selectbox("Fatigue", ["No", "Yes"], help="Persistent tiredness or fatigue")
    
    st.markdown("#### 🫁 Respiratory Symptoms")
    WHEEZING = st.selectbox("Wheezing", ["No", "Yes"], help="Whistling sound while breathing")
    COUGHING = st.selectbox("Coughing", ["No", "Yes"], help="Persistent or chronic cough")
    SHORTNESS_OF_BREATH = st.selectbox("Shortness of Breath", ["No", "Yes"], help="Difficulty breathing")
    SWALLOWING_DIFFICULTY = st.selectbox("Swallowing Difficulty", ["No", "Yes"], help="Trouble swallowing")
    CHEST_PAIN = st.selectbox("Chest Pain", ["No", "Yes"], help="Pain or discomfort in chest")

# Convert yes/no to numeric
def enc(x): return 1 if x == "Yes" else 0

# Build feature dataframe
tabular_data = pd.DataFrame([{
    "AGE": AGE,
    "SMOKING": enc(SMOKING),
    "YELLOW_FINGERS": enc(YELLOW_FINGERS),
    "ANXIETY": enc(ANXIETY),
    "PEER_PRESSURE": enc(PEER_PRESSURE),
    "CHRONIC_DISEASE": enc(CHRONIC_DISEASE),
    "FATIGUE": enc(FATIGUE),
    "ALLERGY": enc(ALLERGY),
    "WHEEZING": enc(WHEEZING),
    "ALCOHOL_CONSUMING": enc(ALCOHOL_CONSUMING),
    "COUGHING": enc(COUGHING),
    "SHORTNESS_OF_BREATH": enc(SHORTNESS_OF_BREATH),
    "SWALLOWING_DIFFICULTY": enc(SWALLOWING_DIFFICULTY),
    "CHEST_PAIN": enc(CHEST_PAIN)
}])

with st.expander("📊 View Input Summary", expanded=False):
    st.dataframe(tabular_data, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ===============================================================
# 🧮 Prediction (Fusion)
# ===============================================================
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("🔍 Analyze Risk Profile", use_container_width=True)

if predict_button:
    try:
        with st.spinner("🧮 Processing health data..."):
            X_scaled = scaler.transform(tabular_data)
            tabular_prob = rf_model.predict_proba(X_scaled)[0, 1]
        
        st.markdown(f"""
        <div class="info-card">
            <h4 style="margin: 0; color: #1976D2;">📊 Clinical Data Analysis Complete</h4>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                Tabular Model Risk Probability: <strong>{tabular_prob:.1%}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Combine both if image uploaded
        if image_prob is not None:
            final_prob = 0.3 * tabular_prob + 0.7 * image_prob
            st.info("🔄 Fusing CT scan and clinical data for comprehensive assessment...")
        else:
            final_prob = tabular_prob
            st.info("ℹ️ Analysis based on clinical data only (no CT scan provided)")

        risk_percent = round(final_prob * 100, 2)

        # ===========================================================
        # 🎯 Display Final Result
        # ===========================================================
        st.markdown("<br>", unsafe_allow_html=True)
        
        if risk_percent >= 75:
            st.markdown(f"""
            <div class="error-box">
                <h2 style="margin: 0; color: #D32F2F;">🚨 High Risk Alert</h2>
                <h1 style="margin: 0.5rem 0; color: #D32F2F; font-size: 3rem;">{risk_percent}%</h1>
                <p style="margin: 0; font-size: 1.2rem; color: #C62828;">
                    <strong>Immediate medical consultation recommended.</strong>
                </p>
                <p style="margin-top: 1rem; color: #B71C1C;">
                    This indicates a significant risk level. Please schedule an appointment with an oncologist 
                    or pulmonologist as soon as possible for comprehensive evaluation and testing.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        elif risk_percent >= 40:
            st.markdown(f"""
            <div class="warning-box">
                <h2 style="margin: 0; color: #F57C00;">⚠️ Moderate Risk Detected</h2>
                <h1 style="margin: 0.5rem 0; color: #F57C00; font-size: 3rem;">{risk_percent}%</h1>
                <p style="margin: 0; font-size: 1.2rem; color: #EF6C00;">
                    <strong>Medical check-up recommended.</strong>
                </p>
                <p style="margin-top: 1rem; color: #E65100;">
                    While not immediately critical, this risk level warrants professional medical evaluation. 
                    Please consult with your healthcare provider to discuss further diagnostic tests.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown(f"""
            <div class="success-box">
                <h2 style="margin: 0; color: #2E7D32;">✅ Low Risk Profile</h2>
                <h1 style="margin: 0.5rem 0; color: #2E7D32; font-size: 3rem;">{risk_percent}%</h1>
                <p style="margin: 0; font-size: 1.2rem; color: #1B5E20;">
                    <strong>Positive health indicators detected.</strong>
                </p>
                <p style="margin-top: 1rem; color: #1B5E20;">
                    The analysis suggests a low probability of lung cancer. Continue maintaining healthy 
                    lifestyle habits and regular health check-ups as preventive measures.
                </p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.markdown(f"""
        <div class="error-box">
            <h3 style="margin: 0; color: #D32F2F;">❌ Analysis Error</h3>
            <p style="margin: 0.5rem 0 0 0;">An error occurred during processing: {str(e)}</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <h4 style="color: #2E7D32; margin-bottom: 0.5rem;">💡 Important Disclaimer</h4>
    <p style="color: #555; margin: 0;">
        This application combines AI analysis from both CT scans and patient health data. 
        It is designed as a preliminary screening tool and <strong>not a substitute for professional medical diagnosis</strong>. 
        Always consult with qualified healthcare professionals for accurate diagnosis and treatment.
    </p>
    <p style="color: #777; margin-top: 1rem; font-size: 0.9rem;">
        Developed with ❤️ by Harsh Verma| Powered by AI & Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)