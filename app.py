import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
MODEL_PATH = "malaria_mobilenetv2_model (2).keras" # ปรับตามไฟล์ล่าสุดของมึงละสสับ
IMG_SIZE = (224, 224)
SAMPLE_DIR = "samples"

# --- PAGE CONFIG ---
st.set_page_config(page_title="MalariaScope · Vertical Space", layout="centered")

# ─── THE SPACESHIP UI (CSS - Mobile & Vertical Optimized) ──────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;500;700&family=Sarabun:wght@300;500;700&display=swap');

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

/* ปรับแต่งความกว้างของช่อง Selectbox */
div[data-testid="stSelectbox"] {
    width: 100% !important;
    max-width: 450px !important;
    margin: 0 auto !important;
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
    max-width: 450px !important;
    margin: 0 auto !important;
}

/* ผลลัพธ์ (Result) */
.result-display {
    background: rgba(255, 255, 255, 0.03);
    padding: 20px;
    text-align: center !important; 
    border: 1px solid rgba(77, 163, 255, 0.2);
    margin-top: 20px;
    width: 100%;
    max-width: 450px !important;
    margin-left: auto !important;
    margin-right: auto !important;
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

# ─── HEADER (เอา 98.5% Precision ออกตามสั่ง) ──────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">Lightweight Image classification for Malaria detection using mobilenetv2</h1>
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

# ─── DATA INPUT SECTION (ดึงรูปจากโฟลเดอร์ samples) ──────────────────────────────────
img = None

if os.path.exists(SAMPLE_DIR):
    files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if files:
        st.markdown('<p style="font-family:Orbitron; font-size:0.8rem; margin-top:10px; text-align:center; color:#8892b0;">CHOOSE REPOSITORY DATASET</p>', unsafe_allow_html=True)
        choice = st.selectbox("", files, label_visibility="collapsed")
        if choice:
            img = Image.open(os.path.join(SAMPLE_DIR, choice)).convert("RGB")
    else:
        st.warning("⚠️ No sample image files found in 'samples' directory.")
else:
    st.error("⚠️ SYSTEM ERROR: 'samples' directory not found.")

# ─── SCANNING & RESULTS ──────────────────────────────────────────────────────
if img:
    st.markdown('<div class="img-container">', unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("START ANALYTICS"):
        # Preprocess รูปภาพส่งเข้าโมเดล
        img_p = img.resize(IMG_SIZE)
        img_arr = image.img_to_array(img_p)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = preprocess_input(img_arr)
        
        with st.spinner("PROCESSING..."):
            # 🛠️ สั่งทำนายผลและดึงคะแนนของทั้ง 3 คลาสออกมาครบถ้วน
            predictions = model.predict(img_arr)[0]
        
        # 🛠️ ใช้ np.argmax หาตำแหน่งคลาสที่ทำคะแนนได้สูงสุดจริงๆ (แก้บั๊กเอ๋อทายคลาส 0 ซ้ำซาก)
        predicted_class = np.argmax(predictions)
        conf = predictions[predicted_class] # ดึงค่าความมั่นใจของคลาสที่ชนะมาแสดงผล
        
        # คัดกรองคลาสแปลงข้อความภาษาไทยพร้อมยัดสีสไตล์อวกาศ
        if predicted_class == 0:
            status = "เซลล์ติดเชื้อ"
            color = "#ff3d6b"  # เจอเชื้อสาดสีแดงกระแทกตา
        else:
            status = "เซลล์ปกติ"
            color = "#00d2b4"  # ปลอดภัยสาดสีเขียวมิ้นต์อวกาศ
        
        # แสดงกล่องสรุปผลวิเคราะห์แนวตั้งตรงกลางจออย่างเท่
        st.markdown(f"""
        <div class="result-display" style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
            <p style="font-family:Rajdhani; color:#8892b0; margin:0; font-size:0.8rem; text-transform:uppercase; width:100%;">SCAN RESULT</p>
            <h2 style="font-family:'Sarabun', 'Orbitron', sans-serif; color:{color}; margin: 5px 0; letter-spacing:1px; font-size: 2rem; font-weight: 700; width:100%;">{status}</h2>
            <div style="margin: 10px auto; height: 1px; background: rgba(77,163,255,0.2); width: 100%;"></div>
            <p style="font-family:Rajdhani; color:#8892b0; margin:0; font-size:0.8rem; text-transform:uppercase; width:100%;">CONFIDENCE LEVEL</p>
            <h1 style="font-family:Orbitron; font-size:1.8rem; margin:0; color:#ffffff; width:100%;">{conf*100:.2f}%</h1>
        </div>
        """, unsafe_allow_html=True)
