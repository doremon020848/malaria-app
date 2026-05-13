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
st.set_page_config(page_title="MalariaScope · Vertical Space", layout="centered")

# ─── THE SPACESHIP UI (CSS - Mobile & Vertical Optimized) ──────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Montserrat:wght@500;600;700;800&display=swap');

#MainMenu, footer, header { visibility: hidden; }

/* หัวข้อใหญ่แนวตั้ง */
.hero-header {
    text-align: center;
    padding: 16px 5%; 
    border-bottom: 1.5px solid rgba(77, 163, 255, 0.2);
    margin-bottom: 1.5rem;
    width: 100%;
}
.hero-title {
    font-family: 'Montserrat', sans-serif; #
    font-size: 32px !important;
    font-weight: 700;
    letter-spacing: 1px;
    background: linear-gradient(180deg, #ffffff, #4da3ff);
    text-transform: uppercase;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: block; 
    line-height: 1.3 !important;
    margin: 0 auto;
}

/* --- แก้ไขปุ่ม Action ให้รองรับมือถือ --- */
div.stButton {
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
    margin-top: 1rem !important;
}

div.stButton > button {
    width: 100% !important;        
    max-width: 280px !important;  
    background: #4da3ff !important;
    color: #02060c !important;
    font-family: 'Montserrat', sans-serif !important;
    height: 3.8rem !important;
    border-radius: 8px !important;
    border: none !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: 0.3s ease-in-out !important;
}

div.stButton > button:hover {
    background: #ffffff !important;
    box-shadow: 0 0 20px rgba(77, 163, 255, 0.6) !important;
    transform: scale(1.02);
}

/* ผลลัพธ์ (Result) */
.result-display {
    background: rgba(255, 255, 255, 0.03);
    padding: 20px;
    border: 1px solid rgba(77, 163, 255, 0.2);
    margin-top: 20px;
    width: 100%;
    /* ใช้ Flexbox คุมไปเลย จบแน่นอน */
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
}

/* ช่อง Preview รูป */
.img-container {
    border: 0px solid rgba(77, 163, 255, 0.2);
    padding: 0px;
    background: rgba(0,0,0,0.5);
    width: 100%;
}

/* =======================================================
                       MOBILE RESPONSIVE
   ======================================================= */
@media (max-width: 768px) {
    .hero-title {
        font-size: 26px !important; 
        letter-spacing: 0px !important;
    }
    .hero-header {
        padding: 1px 0; 
    }
    .result-display {
        padding: 15px; 
    }
}
</style>
""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">Lightweight Image classification for Malaria detection using mobilenetv2</h1>
    <p style="font-family:'Inter', sans-serif; color:#4da3ff; letter-spacing:1px; margin-top:10px; font-weight: 500;">with 98.5% Precision</p>
</div>
""", unsafe_allow_html=True)

# ─── MODEL LOADING ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    model = load_my_model()
except Exception as e:
    st.error(f"🚀 SYSTEM ERROR: MODEL_NOT_FOUND")
    st.stop()

# ─── DATA INPUT SECTION ──────────────────────────────────────────────────────
st.markdown('<p style="font-family:"Inter", sans-serif; font-size:1.2rem; margin-top:0px; font-weight:600;">SELECTION MODE</p>', unsafe_allow_html=True)
mode = st.radio("", ["SAMPLES", "UPLOAD"], horizontal=True, label_visibility="collapsed")

img = None
if mode == "SAMPLES":
    if os.path.exists(SAMPLE_DIR):
        files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if files:
            st.markdown('<p style="font-family:"Inter", sans-serif; font-size:1.2rem; font-weight:600; margin-bottom:-10px;">CHOOSE DATASET</p>', unsafe_allow_html=True)
            choice = st.selectbox("", files, label_visibility="collapsed")
            if choice:
                img = Image.open(os.path.join(SAMPLE_DIR, choice)).convert("RGB")
        else:
            st.warning("No samples found in directory.")
else:
    st.markdown('<p style="font-family:"Inter", sans-serif; font-size:1.2rem; font-weight:600; margin-bottom:-10px;">UPLOAD CELL DATA</p>', unsafe_allow_html=True)
    up = st.file_uploader("", type=["jpg", "png"], label_visibility='collapsed')
    if up: img = Image.open(up).convert("RGB")

# ─── SCANNING & RESULTS ──────────────────────────────────────────────────────
if img:
    st.markdown('<div class="img-container">', unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("START ANALYTICS"):
        # Preprocess
        img_p = img.resize(IMG_SIZE)
        img_arr = image.img_to_array(img_p)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = preprocess_input(img_arr)
        
        with st.spinner("PROCESSING..."):
            
            pred = float(model.predict(img_arr)[0][0])
            
            is_safe = pred > 0.5
            conf = pred if is_safe else 1 - pred
            color = "#00d2b4" if is_safe else "#ff3d6b"
            status = "NORMAL CELL" if is_safe else "INFECTED DETECTED"
            
            st.markdown(f"""
                <div style="text-align: left; background: #0e1117; padding: 20px; border-radius: 10px;">
                    <p style="font-family:'Inter', sans-serif; color:#8892b0; margin:0; font-size:0.8rem; text-transform:uppercase; font-weight:500;">SCAN RESULT</p>
                    <h2 style="font-family:'Inter', sans-serif; color:{color}; margin: 5px 0; letter-spacing:1px; font-size: 1.5rem; font-weight:700;">{status}</h2>
                    <p style="font-family:'Inter', sans-serif; color:#8892b0; margin:0; font-size:0.8rem; text-transform:uppercase; font-weight:500;">CONFIDENCE LEVEL</p>
                    <h1 style="font-family:'Inter', sans-serif; font-size:1.8rem; margin:0; color:#ffffff; font-weight:700;">{conf*100:.2f}%</h1>
                </div>
            """, unsafe_allow_html=True)
