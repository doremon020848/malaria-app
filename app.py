import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
MODEL_PATH = "best_model (6).h5"
IMG_SIZE = (224, 224)
SAMPLE_DIR = "samples"

# เกณฑ์ความมั่นใจขั้นต่ำ (ถ้าต่ำกว่า 60% ให้ถือว่า "ไม่ใช่" ภาพที่เข้าข่าย)
CONFIDENCE_THRESHOLD = 0.60 

# --- PAGE CONFIG ---
st.set_page_config(page_title="Malaria Detection", layout="centered")

# ─── UI (CSS - Mobile & Vertical Optimized) ──────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Montserrat:wght@500;600;700;800&display=swap');

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

.hero-header {
    text-align: center;
    padding: 16px 0; 
    border-bottom: 1.5px solid rgba(77, 163, 255, 0.2);
    margin-bottom: 1.5rem;
    width: 100%;
}
.hero-header a {
    display: none !important;
}

.hero-title {
    font-family: 'Montserrat', sans-serif;
    font-size: clamp(22px, 6vw, 36px) !important; 
    font-weight: 700;
    letter-spacing: 1px;
    background: linear-gradient(180deg, #ffffff, #4da3ff);
    text-transform: uppercase;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: block; 
    line-height: 1.3 !important;
    margin: 0 auto;
    word-wrap: break-word;
    overflow-wrap: break-word;
    pointer-events: none;
}

.hero-subtitle {
    font-family: 'Inter', sans-serif !important;
    color: #67A8E6 !important;
    letter-spacing: 1px !important;
    margin-top: 0px !important;
    font-weight: 400 !important;
    font-size: 12px !important; 
    text-align: center !important;
}

/* --- แก้ไขปุ่ม Action --- */
div.stButton {
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
    margin-top: 1rem !important;
}

div.stButton > button {
    width: 100% !important;        
    max-width: 320px !important;
    background: #4da3ff !important;
    color: #02060c !important;
    font-family: 'Montserrat', sans-serif !important;
    height: 3.5rem !important;
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

.result-display {
    background: rgba(255, 255, 255, 0.05) !important;
    padding: 20px;
    text-align: left !important; 
    border: 1px solid rgba(77, 163, 255, 0.2); 
    margin-top: 30px;
    border-radius: 10px; 
    width: 100%;
}

/* =======================================================
                    MOBILE RESPONSIVE
   ======================================================= */
@media (max-width: 768px) {
    .hero-title {
        letter-spacing: 0px !important;
    }
    .hero-header {
        padding: 5px 0; 
    }
    .result-display {
        padding: 15px; 
        margin-top: 20px;
    }
    div.stButton > button {
        height: 3.2rem !important;
        font-size: 1rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">Lightweight Image classification for<br>Malaria detection<br>using mobilenetv2</h1>
    <p class="hero-subtitle">with 98.5% Precision</p>
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
st.markdown('<p style="font-family:\'Inter\', sans-serif; font-size:1.2rem; margin-top:0px; font-weight:600;">SELECTION MODE</p>', unsafe_allow_html=True)
mode = st.radio("", ["SAMPLES", "UPLOAD"], horizontal=True, label_visibility="collapsed")

img = None
if mode == "SAMPLES":
    if os.path.exists(SAMPLE_DIR):
        files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if files:
            st.markdown('<p style="font-family:\'Inter\', sans-serif; font-size:1.2rem; font-weight:600; margin-bottom:-10px;">CHOOSE DATASET</p>', unsafe_allow_html=True)
            choice = st.selectbox("", files, label_visibility="collapsed")
            if choice:
                img = Image.open(os.path.join(SAMPLE_DIR, choice)).convert("RGB")
        else:
            st.warning("No samples found in directory.")
else:
    st.markdown('<p style="font-family:\'Inter\', sans-serif; font-size:1.2rem; font-weight:600; margin-bottom:-10px;">UPLOAD CELL DATA</p>', unsafe_allow_html=True)
    up = st.file_uploader("", type=["jpg", "png"], label_visibility='collapsed')
    if up: img = Image.open(up).convert("RGB")

# ─── SCANNING & RESULTS ──────────────────────────────────────────────────────
if img:
    st.markdown('', unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    
    if st.button("START ANALYTICS"):
        # Preprocess
        img_p = img.resize(IMG_SIZE)
        img_arr = image.img_to_array(img_p)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = preprocess_input(img_arr)
        
with st.spinner("PROCESSING..."):
            
            # --- 1. ทำนายผลแบบใหม่ (3 คลาส) ---
            predictions = model.predict(img_arr)
            
            # ⚠️ สำคัญมาก: มึงต้องเรียงชื่อคลาสให้ตรงกับตอนที่มึง Train โมเดลนะ 
            # ถ้ามึงใช้ image_dataset_from_directory มันจะเรียงตามตัวอักษรโฟลเดอร์ (malaria, normal, others)
            class_names = ['malaria', 'normal', 'others'] 
            
            max_confidence = np.max(predictions[0])
            predicted_class_index = np.argmax(predictions[0])
            
            # --- 2. ดักจับภาพด้วย Threshold ---
            # ใช้ CONFIDENCE_THRESHOLD 0.60 จากที่มึงตั้งไว้ข้างบนไฟล์ (หรือจะแก้เป็น 0.80 ก็ไปแก้ข้างบน)
            if max_confidence < CONFIDENCE_THRESHOLD:
                # ถ้าความมั่นใจต่ำกว่าเกณฑ์ ให้เตะลง Others ทันที
                status = "NOT BLOOD CELL (LOW CONFIDENCE)"
                conf = max_confidence  # โชว์เปอเซ็นต์ที่มันเดาได้ไปเลย
                color = "#ff9f43"  # สีส้ม Warning
            else:
                # ถ้ามั่นใจเกินเกณฑ์ ก็มาดูว่ามันคือคลาสไหน
                final_class = class_names[predicted_class_index]
                conf = max_confidence
                
                if final_class == 'normal':
                    status = "NORMAL CELL"
                    color = "#00d2b4"  # สีเขียว
                elif final_class == 'malaria':
                    status = "INFECTED DETECTED"
                    color = "#ff3d6b"  # สีแดง
                elif final_class == 'others': 
                    # กรณีที่มันมั่นใจทะลุเกณฑ์ว่าเป็นภาพขยะ (Others)
                    status = "NOT BLOOD CELL (OTHERS)"
                    color = "#ff9f43"  # สีส้ม
            
            # --- 3. ส่วนแสดงผล UI ของมึงเหมือนเดิม ไม่ต้องแก้ ---
            st.markdown(f"""
                <div class="result-display">
                    <p style="font-family:'Inter', sans-serif; color:#8892b0; margin:0; font-size:0.8rem; text-transform:uppercase; font-weight:500;">SCAN RESULT</p>
                    <h2 style="font-family:'Inter', sans-serif; color:{color}; margin: 5px 0; letter-spacing:1px; font-size: 1.5rem; font-weight:700;">{status}</h2>
                    <p style="font-family:'Inter', sans-serif; color:#8892b0; margin:0; font-size:0.8rem; text-transform:uppercase; font-weight:500;">CONFIDENCE LEVEL</p>
                    <h1 style="font-family:'Inter', sans-serif; font-size:1.8rem; margin:0; color:#ffffff; font-weight:700;">{conf*100:.2f}%</h1>
                </div>
            """, unsafe_allow_html=True)
