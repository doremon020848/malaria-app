import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
MODEL_PATH = "best_model_lite.h5" 
IMG_SIZE = (224, 224)
SAMPLE_DIR = "samples"

# --- PAGE CONFIG ---
st.set_page_config(page_title="MalariaScope · Space Edition", layout="wide")

# ─── THE SPACESHIP UI (CSS) ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;500;700&display=swap');

/* พื้นหลังอวกาศ */
.stApp {
    background: radial-gradient(circle at top right, #0a192f 0%, #02060c 100%);
    color: #e6f1ff;
}

/* ลบ Chrome ของ Streamlit */
#MainMenu, footer, header { visibility: hidden; }

/* หัวข้อแนว Sci-Fi */
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 3.5rem;
    font-weight: 700;
    letter-spacing: 5px;
    text-transform: uppercase;
    background: linear-gradient(180deg, #ffffff 0%, #4da3ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
}

.hero-subtitle {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.2rem;
    color: #8892b0;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* กล่องข้อมูลแนว Grid ในรูป (Cards) */
.info-card {
    background: rgba(10, 25, 47, 0.7);
    border: 1px solid rgba(77, 163, 255, 0.2);
    border-radius: 5px;
    padding: 20px;
    text-align: center;
    transition: 0.3s;
}
.info-card:hover {
    border-color: #4da3ff;
    box-shadow: 0 0 15px rgba(77, 163, 255, 0.3);
}

/* ปุ่มแบบในรูป (Blue Button) */
div.stButton > button {
    background: #4da3ff !important;
    color: #02060c !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-radius: 4px !important;
    border: none !important;
    padding: 10px 30px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* แถบเส้นประแยก Section */
.section-line {
    border-top: 1px solid rgba(77, 163, 255, 0.3);
    margin: 40px 0;
}

/* สำหรับช่อง Input */
.stSelectbox, .stRadio, .stFileUploader {
    background: rgba(10, 25, 47, 0.5);
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">MALARIASCAPE</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Advanced Neural Scanning System · V2.0</p>', unsafe_allow_html=True)

# ─── TOP INFO CARDS (เหมือนในรูปมึงเป๊ะ) ──────────────────────────────────────────
col_c1, col_c2, col_c3 = st.columns(3)
with col_c1:
    st.markdown('<div class="info-card"><h3>CORE</h3><p>MobileNetV2</p><small>Neural Backbone</small></div>', unsafe_allow_html=True)
with col_c2:
    st.markdown('<div class="info-card"><h3>PRECISION</h3><p>98.5%</p><small>Validation AUC</small></div>', unsafe_allow_html=True)
with col_c3:
    st.markdown('<div class="info-card"><h3>OBJECT</h3><p>RBC</p><small>Target Analysis</small></div>', unsafe_allow_html=True)

st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

# ─── MODEL LOADING ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    model = load_my_model()
except:
    st.error("🚀 MISSION ABORTED: Model not found.")
    st.stop()

# ─── MAIN SYSTEM (UPLOAD/SELECT) ──────────────────────────────────────────────
col_main, col_res = st.columns([6, 4], gap="large")

with col_main:
    st.markdown('<h3 style="font-family:Orbitron; font-size:1rem;">🛰️ DATA INPUT</h3>', unsafe_allow_html=True)
    mode = st.radio("", ["SAMPLES", "UPLOAD"], horizontal=True)
    
    img = None
    if mode == "SAMPLES":
        if os.path.exists(SAMPLE_DIR):
            files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            choice = st.selectbox("Select Satellite Image:", files)
            if choice:
                img = Image.open(os.path.join(SAMPLE_DIR, choice)).convert("RGB")
    else:
        up = st.file_uploader("Upload New Cell Data:", type=["jpg", "png"])
        if up: img = Image.open(up).convert("RGB")

    if img:
        st.image(img, use_container_width=True)

with col_res:
    st.markdown('<h3 style="font-family:Orbitron; font-size:1rem;">📡 ANALYSIS</h3>', unsafe_allow_html=True)
    if img:
        # Preprocess
        img_p = img.resize(IMG_SIZE)
        img_arr = image.img_to_array(img_p)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = preprocess_input(img_arr)
        
        if st.button("RUN SCANNER"):
            with st.spinner("SCANNING..."):
                pred = float(model.predict(img_arr)[0][0])
            
            is_safe = pred > 0.5
            conf = pred if is_safe else 1 - pred
            
            color = "#00d2b4" if is_safe else "#ff3d6b"
            status = "CLEAN" if is_safe else "INFECTED"
            
            st.markdown(f"""
            <div style="border-left: 5px solid {color}; padding-left: 20px; background: rgba(255,255,255,0.05); padding: 20px;">
                <p style="color:#8892b0; margin:0;">SCAN STATUS</p>
                <h2 style="color:{color}; font-family:Orbitron; margin:0;">{status}</h2>
                <hr style="opacity:0.2">
                <p style="color:#8892b0; margin:0;">CONFIDENCE</p>
                <h2 style="font-family:Orbitron;">{conf*100:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; font-family:Rajdhani; color:#4d7a99;">Technology · Created by ไอ้ยี่สิบ · Web Design ©2026 · Supported by A-SPACE</p>', unsafe_allow_html=True)
