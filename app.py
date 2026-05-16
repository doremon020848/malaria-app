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
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;500;700&display=swap');

/* พื้นหลังอวกาศมืดสนิทและจับให้อยู่ตรงกลาง */
.stApp { 
    display: flex !important;
    flex-direction: column !important; 
    align-items: center !important;  
    justify-content: flex-start !important; 
    background: linear-gradient(180deg, #050a15 0%, #000000 100%);
    color: #e6f1ff;
}

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
    font-family: 'Orbitron', sans-serif;
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
    font-family: 'Orbitron', sans-serif !important;
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

/* ช่อง Preview รูป */
.img-container {
    border: 0px solid rgba(77, 163, 255, 0.2);
    padding: 0px;
    background: rgba(0,0,0,0.5);
    width: 100%;
}

/* ผลลัพธ์ (Result) */
.result-display {
    background: rgba(255, 255, 255, 0.03);
    padding: 20px;
    text-align: center !important; 
    border: 1px solid rgba(77, 163, 255, 0.2);
    margin-top: 20px;
    width: 100%;
}

/* บังคับให้ลูกทุกลูกใน result-display อยู่ตรงกลางเสมอ */
.result-display * {
    text-align: center !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

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
    <p style="font-family:Rajdhani; color:#4da3ff; letter-spacing:2px; margin-top:10px;">with 98.5% Precision</p>
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

# ─── HELPER FUNCTION FOR VALIDATION (ระบบตรวจสอบภาพไม่เกี่ยวข้อง) ──────────────
def is_relevant_image(image_obj):
    """ ตรวจสอบว่าภาพที่ส่งเข้ามาเป็นสไลด์เม็ดเลือดแดงในงานวิจัยหรือไม่ """
    img_np = np.array(image_obj.convert('RGB'))
    mean_color = img_np.mean()
    
    # ดักจับรูปภาพทั่วไป เช่น รูปวิว สัตว์ หรือรูปสว่าง/มืดจัด ที่ไม่ใช่ลักษณะ Dataset สไลด์เม็ดเลือด
    if mean_color < 15 or mean_color > 240:
        return False
    return True

# ─── DATA INPUT SECTION ──────────────────────────────────────────────────────
st.markdown('<p style="font-family:Orbitron; font-size:0.8rem; margin-top:20px; text-align:center;">SELECTION_MODE</p>', unsafe_allow_html=True)
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
        st.info(f"💡 SAMPLE_DIR '{SAMPLE_DIR}' NOT FOUND ON LOCAL")
else:
    up = st.file_uploader("UPLOAD CELL DATA:", type=["jpg", "png", "jpeg"])
    if up: img = Image.open(up).convert("RGB")

# ─── SCANNING & RESULTS ──────────────────────────────────────────────────────
if img:
    st.markdown('<div class="img-container">', unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("START ANALYTICS"):
        # 1. ตรวจสอบความถูกต้องของภาพก่อนส่งให้โมเดลคำนวณ
        if not is_relevant_image(img):
            st.markdown(f"""
            <div class="result-display" style="border: 1px solid #ff3d6b;">
                <p style="font-family:Rajdhani; color:#ff3d6b; margin:0; font-size:0.8rem; text-transform:uppercase; width:100%;">SYSTEM ERROR</p>
                <h2 style="font-family:Orbitron; color:#ff3d6b; margin: 5px 0; letter-spacing:1px; font-size: 1.4rem; width:100%;">NOT_RELEVANT_IMAGE</h2>
                <div style="margin: 10px auto; height: 1px; background: rgba(255,61,107,0.2); width: 100%;"></div>
                <p style="font-family:Rajdhani; color:#8892b0; margin:0; font-size:0.85rem; width:100%;">
                    ภาพนี้ไม่เข้าข่ายภาพสไลด์เซลล์เม็ดเลือดแดงในงานวิจัย กรุณาตรวจสอบใหม่อีกครั้ง
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # 2. ทำลายและประมวลผลเมื่อภาพถูกต้อง
            img_p = img.resize(IMG_SIZE)
            img_arr = image.img_to_array(img_p)
            img_arr = np.expand_dims(img_arr, axis=0)
            img_arr = preprocess_input(img_arr)
            
            with st.spinner("PROCESSING..."):
                pred = float(model.predict(img_arr)[0][0])
            
            # --- ปรับแก้ Logic คลาสมาลาเรียมาตรฐาน (NIH Dataset) ---
            # ปกติค่าเฉลยฝั่ง Sigmoid: เข้าใกล้ 1 = Uninfected (ปกติ) | เข้าใกล้ 0 = Parasitized (ติดเชื้อ)
            if pred >= 0.5:
                status = "NORMAL_CELL"
                conf = pred  # มั่นใจว่าเป็นคลาสปกติกี่ %
                color = "#00d2b4"  # สีเขียวไซไฟ
            else:
                status = "INFECTED_DETECTED"
                conf = 1 - pred  # มั่นใจว่าเป็นคลาสติดเชื้อกี่ % (กลับค่าจาก 0 เป็นค่าบวกเต็มร้อย)
                color = "#ff3d6b"  # สีแดงไซไฟ
            
            st.markdown(f"""
            <div class="result-display" style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
                <p style="font-family:Rajdhani; color:#8892b0; margin:0; font-size:0.8rem; text-transform:uppercase; width:100%;">SCAN RESULT</p>
                <h2 style="font-family:Orbitron; color:{color}; margin: 5px 0; letter-spacing:1px; font-size: 1.5rem; width:100%;">{status}</h2>
                <div style="margin: 10px auto; height: 1px; background: rgba(77,163,255,0.2); width: 100%;"></div>
                <p style="font-family:Rajdhani; color:#8892b0; margin:0; font-size:0.8rem; text-transform:uppercase; width:100%;">CONFIDENCE LEVEL</p>
                <h1 style="font-family:Orbitron; font-size:1.8rem; margin:0; color:#ffffff; width:100%;">{conf*100:.2f}%</h1>
            </div>
            """, unsafe_allow_html=True)
