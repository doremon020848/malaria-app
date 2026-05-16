import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
MODEL_PATH = "best_model_lite (1).h5"  # ชี้ไปที่ไฟล์โมเดล 3 คลาสของมึง
IMG_SIZE = (224, 224)
SAMPLE_DIR = "samples"

# เกณฑ์ความมั่นใจขั้นต่ำ (ถ้าคะแนนคลาสที่ชนะไม่ถึง 60% ให้ตบเข้า others ทันที)
CONFIDENCE_THRESHOLD = 0.60 

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

/* กล่อง Info Card แบบเรียงแถวเดียว */
.info-card-vertical {
    background: rgba(10, 25, 47, 0.6);
    border: 1px solid rgba(77, 163, 255, 0.15);
    border-left: 4px solid #4da3ff;
    padding: 15px; 
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
}
.info-label { font-family: 'Rajdhani'; color: #8892b0; text-transform: uppercase; font-size: 0.8rem; }
.info-value { font-family: 'Orbitron'; color: #4da3ff; font-size: 0.9rem; }

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

/* =======================================================
   MOBILE RESPONSIVE (ถ้าหน้าจอเล็กกว่า 768px ให้ทำตามนี้)
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
    <p style="font-family:Rajdhani; color:#4da3ff; letter-spacing:2px; margin-top:10px;">with 3-Class Spaceship Robust System</p>
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

# ─── SIDEBAR / DEBUG CONTROLS ────────────────────────────────────────────────
st.sidebar.header("🛠️ SPACE MONITOR")
CONFIDENCE_THRESHOLD = st.sidebar.slider("เกณฑ์การดักภาพมั่ว (Threshold)", 0.0, 1.0, 0.60, step=0.05)
show_debug = st.sidebar.checkbox("เปิดดู Debug หลังบ้าน", value=False)

# ─── DATA INPUT SECTION ──────────────────────────────────────────────────────
st.markdown('<p style="font-family:Orbitron; font-size:0.8rem; margin-top:20px; text-align:center;">SELECTION_MODE</p>', unsafe_allow_html=True)
mode = st.radio("", ["SAMPLES", "UPLOAD"], horizontal=True, label_visibility="collapsed")

img = None
if mode == "SAMPLES":
    if os.path.exists(SAMPLE_DIR):
        files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if files:
            st.markdown('<p style="font-family:Rajdhani; font-size:0.9rem; margin-bottom:5px;">CHOOSE DATASET:</p>', unsafe_allow_html=True)
            choice = st.selectbox("", files, label_visibility="collapsed")
            if choice:
                img = Image.open(os.path.join(SAMPLE_DIR, choice)).convert("RGB")
        else:
            st.warning("No samples found in directory.")
else:
    st.markdown('<p style="font-family:Rajdhani; font-size:0.9rem; margin-bottom:5px;">UPLOAD CELL DATA:</p>', unsafe_allow_html=True)
    up = st.file_uploader("", type=["jpg", "png"], label_visibility="collapsed")
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
            # 1. ทำนายผลและครอบด้วย Softmax ป้องกันภาพหลุดหลงคลาส
            raw_predictions = model.predict(img_arr)
            predictions = tf.nn.softmax(raw_predictions).numpy()
            
            # ลำดับคลาสโมเดล 3 คลาสของมึง
            class_names = ['malaria', 'normal', 'others'] 
            
            max_confidence = float(np.max(predictions[0]))
            predicted_class_index = np.argmax(predictions[0])
            final_class = class_names[predicted_class_index]
            
            # เปิดส่องสถิติถ้าติ๊กถูกที่ Sidebar
            if show_debug:
                st.sidebar.subheader("📊 ข้อมูลวิเคราะห์")
                st.sidebar.write(f"Raw Output: {raw_predictions[0]}")
                st.sidebar.write(f"Softmax Prob: {predictions[0]}")
                st.sidebar.write(f"Class Index: {predicted_class_index} ({final_class})")
            
            # 2. คัดกรองผ่านเกณฑ์เงื่อนไขควบคุม (Threshold)
            if max_confidence < CONFIDENCE_THRESHOLD:
                status = "NOT_BLOOD_CELL"
                color = "#ff9f43"  # สีส้มแจ้งเตือนภาพมั่ว/ไม่มั่นใจ
                conf = max_confidence
            else:
                conf = max_confidence
                if final_class == 'normal':
                    status = "NORMAL_CELL"
                    color = "#00d2b4"  # สีเขียวเซฟ
                elif final_class == 'malaria':
                    status = "INFECTED_DETECTED"
                    color = "#ff3d6b"  # สีแดงอันตราย
                else:
                    status = "NOT_BLOOD_CELL"
                    color = "#ff9f43"  # สีส้มคลาส others
            
            # 3. แสดงหน้าต่างดีไซน์อวกาศแบบจัดกึ่งกลาง
            st.markdown(f"""
            <div class="result-display" style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
                <p style="font-family:Rajdhani; color:#8892b0; margin:0; font-size:0.8rem; text-transform:uppercase; width:100%;">SCAN RESULT</p>
                <h2 style="font-family:Orbitron; color:{color}; margin: 5px 0; letter-spacing:1px; font-size: 1.5rem; width:100%;">{status}</h2>
                <div style="margin: 10px auto; height: 1px; background: rgba(77,163,255,0.2); width: 100%;"></div>
                <p style="font-family:Rajdhani; color:#8892b0; margin:0; font-size:0.8rem; text-transform:uppercase; width:100%;">CONFIDENCE LEVEL</p>
                <h1 style="font-family:Orbitron; font-size:1.8rem; margin:0; color:#ffffff; width:100%;">{conf*100:.2f}%</h1>
            </div>
            """, unsafe_allow_html=True)
