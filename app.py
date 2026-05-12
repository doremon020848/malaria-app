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
st.set_page_config(page_title="Malaria Detection", layout="centered")

# ─── CLEAN BLUE UI (CSS) ──────────────────────────────
st.markdown("""
<style>
/* พื้นหลังสีฟ้าอ่อนแบบคลีนๆ */
.stApp {
    background-color: #f0f7ff;
}

/* ซ่อน Header/Footer ของ Streamlit */
#MainMenu, footer, header { visibility: hidden; }

/* กล่อง Header ชื่อโปรเจกต์ */
.title-box {
    background-color: #ffffff;
    padding: 20px;
    text-align: center;
    border: 1px solid #d0e3f5;
    margin-bottom: 25px;
    box-shadow: 0px 2px 5px rgba(0,0,0,0.05);
}
.title-text {
    font-family: sans-serif;
    color: #0f3a61;
    font-size: 1.4rem;
    font-weight: 600;
    margin: 0;
}

/* หัวข้อ Label เล็กๆ เหนือกล่อง */
.section-label {
    font-family: sans-serif;
    color: #2b6cb0;
    font-size: 0.95rem;
    margin-bottom: 5px;
    font-weight: 500;
}

/* กล่องขาวสำหรับโชว์ผลลัพธ์ */
.white-box {
    background-color: #ffffff;
    border: 1px solid #d0e3f5;
    padding: 15px;
    color: #2d3748;
    font-family: sans-serif;
    font-size: 1.1rem;
    min-height: 50px;
    margin-bottom: 15px;
    box-shadow: 0px 1px 3px rgba(0,0,0,0.02);
}

/* ปุ่ม Analysis */
div.stButton > button {
    background-color: #ffffff !important;
    color: #2b6cb0 !important;
    border: 1px solid #2b6cb0 !important;
    font-family: sans-serif !important;
    height: 3rem !important;
    border-radius: 4px !important;
    font-weight: bold !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: 0.3s;
}
div.stButton > button:hover {
    background-color: #2b6cb0 !important;
    color: #ffffff !important;
}

/* ปรับสีตัวอักษรของ Radio/Selectbox ให้เข้ากับพื้นสว่าง */
.stRadio > label, .stSelectbox > label, .stFileUploader > label {
    color: #2d3748 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-box">
    <p class="title-text">Lightweight Image Classification<br>for Malaria Detection using MobileNetV2</p>
</div>
""", unsafe_allow_html=True)

# ─── MODEL LOADING ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    model = load_my_model()
except:
    st.error("⚠️ SYSTEM ERROR: ไม่พบไฟล์โมเดล (MODEL_NOT_FOUND)")
    st.stop()

# ─── DATA INPUT SECTION ──────────────────────────────────────────────────────
st.markdown('<p class="section-label">Selection mode</p>', unsafe_allow_html=True)
mode = st.radio("", ["Samples", "Upload"], horizontal=True, label_visibility="collapsed")

img = None
if mode == "Samples":
    st.markdown('<p class="section-label">Choose Dataset</p>', unsafe_allow_html=True)
    if os.path.exists(SAMPLE_DIR):
        files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        choice = st.selectbox("", files, label_visibility="collapsed")
        if choice:
            img = Image.open(os.path.join(SAMPLE_DIR, choice)).convert("RGB")
    else:
        st.warning("ไม่พบโฟลเดอร์ samples")
else:
    st.markdown('<p class="section-label">Choose Dataset</p>', unsafe_allow_html=True)
    up = st.file_uploader("", type=["jpg", "png"], label_visibility="collapsed")
    if up: img = Image.open(up).convert("RGB")

# ─── IMAGE PREVIEW ───────────────────────────────────────────────────────────
if img:
    # โชว์รูปเหมือนในกรอบใหญ่ๆ ของเทมเพลต
    st.markdown('<div style="background-color: white; padding: 10px; border: 1px solid #d0e3f5; margin-bottom: 20px;">', unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ปุ่ม Analysis (จัดให้อยู่ตรงกลางตามรูป)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_btn = st.button("Analysis")

    # ─── SCANNING & RESULTS ──────────────────────────────────────────────────────
    if analyze_btn:
        # Preprocess
        img_p = img.resize(IMG_SIZE)
        img_arr = image.img_to_array(img_p)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = preprocess_input(img_arr)
        
        with st.spinner("Analyzing..."):
            pred = float(model.predict(img_arr)[0][0])
        
        is_safe = pred > 0.5
        conf = pred if is_safe else 1 - pred
        
        # กำหนดข้อความและสีให้เข้ากับโทน
        if is_safe:
            status = "Normal"
            status_color = "#2f855a" # สีเขียว
        else:
            status = "Malaria Infected"
            status_color = "#c53030" # สีแดง
            
        # Analysis result box
        st.markdown('<p class="section-label">Analysis result</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="white-box" style="color: {status_color}; font-weight: bold;">{status}</div>', unsafe_allow_html=True)
        
        # Confidence Level box
        st.markdown('<p class="section-label">Confidence Level</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="white-box">{conf*100:.2f}%</div>', unsafe_allow_html=True)
        
        # Grad-CAM box (รอใส่ฟังก์ชันจริง)
        st.markdown('<p class="section-label">Grad-CAM</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="white-box" style="height: 250px; display: flex; align-items: center; justify-content: center; background-color: #f7fafc; color: #a0aec0; border: 1px dashed #cbd5e0;">
            [ Grad-CAM Visualization Will Appear Here ]
        </div>
        """, unsafe_allow_html=True)
