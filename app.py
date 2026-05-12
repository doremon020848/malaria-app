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

# ─── THE SPACESHIP UI (CSS - Vertical Optimized) ──────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;500;700&display=swap');

/* พื้นหลังอวกาศมืดสนิท */
.stApp {
    background: linear-gradient(180deg, #050a15 0%, #000000 100%);
    color: #e6f1ff;
}

#MainMenu, footer, header { visibility: hidden; }

/* หัวข้อใหญ่แนวตั้ง */
.hero-header {
    text-align: center;
    padding: 2rem 0;
    border-bottom: 1.5px solid rgba(77, 163, 255, 0.2);
    margin-bottom: 2rem;
}
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 32px !important; /* ลองปรับเลขนี้ดูครับ 32px, 40px, 48px */
    font-weight: 700;
    letter-spacing: 1px; /* ลดระยะห่างตัวอักษรลงหน่อยเพราะชื่อยาว เดี๋ยวล้นจอ */
    background: linear-gradient(180deg, #ffffff, #4da3ff);
    text-transform: uppercase;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: block; 
    line-height: 1.3 !important;
    margin: 0 auto;
}
/* กล่อง Info Card แบบเรียงแถวเดียว (แนวตั้ง) */
.info-card-vertical {
    background: rgba(10, 25, 47, 0.6);
    border: 1px solid rgba(77, 163, 255, 0.15);
    border-left: 4px solid #4da3ff;
    padding: 15px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.info-label { font-family: 'Rajdhani'; color: #8892b0; text-transform: uppercase; font-size: 0.8rem; }
.info-value { font-family: 'Orbitron'; color: #4da3ff; font-size: 1rem; }

/* ปุ่ม Action */
div.stButton > button {
    width: 1000% !important;
    background: #4da3ff !important;
    color: #02060c !important;
    font-family: 'Orbitron', sans-serif !important;
    height: 3.5rem !important;
    border-radius: 5 !important;
    border: none !important;
    font-size: 1.1rem !important;
    margin-top: 1rem;
}

div.stButton > button:hover {
    background: #ffffff !important;
    box-shadow: 0 0 20px rgba(77, 163, 255, 0.6) !important;
    transform: scale(1.01); /* ขยายใหญ่ขึ้นนิดนึงเวลาจ่อเมาส์ */
}

/* ช่อง Preview รูป */
.img-container {
    border: 0px solid rgba(77, 163, 255, 0.2);
    padding: 0px;
    background: rgba(0,0,0,0.5);
}

/* ผลลัพธ์ (Result) */
.result-display {
    background: rgba(255, 255, 255, 0.03);
    padding: 25px;
    text-align: center;
    border: 1px solid rgba(77, 163, 255, 0.2);
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">Lightweight Image classification for Malaria detection using mobilenetv2</h1>
    <p style="font-family:Rajdhani; color:#4da3ff; letter-spacing:2px;">with 98.5% Precision</p>
</div>
""",unsafe_allow_html=True)

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
st.markdown('<p style="font-family:Orbitron; font-size:0.8rem; margin-top:20px;">SELECTION_MODE</p>', unsafe_allow_html=True)
mode = st.radio("", ["SAMPLES", "UPLOAD"], horizontal=True, label_visibility="collapsed")

img = None
if mode == "SAMPLES":
    if os.path.exists(SAMPLE_DIR):
        files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if files:
            choice = st.selectbox("CHOOSE DATASET:", files)
            if choice:
                img = Image.open(os.path.join(SAMPLE_DIR, choice)).convert("RGB")
        else:
            st.warning("No samples found in directory.")
else:
    up = st.file_uploader("UPLOAD CELL DATA:", type=["jpg", "png"])
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
        status = "NORMAL_CELL" if is_safe else "INFECTED_DETECTED"
        
        st.markdown(f"""
        <div class="result-display">
            <p style="font-family:Rajdhani; color:#8892b0; margin:0;">SCAN RESULT</p>
            <h2 style="font-family:Orbitron; color:{color}; margin:0; letter-spacing:2px;">{status}</h2>
            <div style="margin: 15px 0; height: 1px; background: rgba(77,163,255,0.2);"></div>
            <p style="font-family:Rajdhani; color:#8892b0; margin:0;">CONFIDENCE LEVEL</p>
            <h1 style="font-family:Orbitron; font-size:2.5rem; margin:0;">{conf*100:.2f}%</h1>
        </div>
        """, unsafe_allow_html=True)
