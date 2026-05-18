import os
import zipfile
import json
import requests
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from io import BytesIO

# --- CONFIGURATION ---
MODEL_PATH = "best_model (6).keras"  # ใช้โมเดลตัวล่าสุดที่มึงให้มา
FIXED_MODEL_PATH = "fixed_spaceship_model.keras"
IMG_SIZE = (224, 224)

# ----------------------------------------------------
# 1. ตั้งค่าลิงก์ GitHub Repository (แก้ไขชื่อ User/Repo ตรงนี้เลยสัส)
# ----------------------------------------------------
GITHUB_RAW_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/samples/"

# รายชื่อไฟล์รูปภาพบน GitHub โฟลเดอร์ samples ของมึง
SAMPLE_IMAGES = [
    "CHOOSE_DATASET...",  # ตัวเลือกแรกให้แสดงคำเท่ๆ
    "cell_infected_1.png",
    "cell_infected_2.png",
    "cell_uninfected_1.png",
    "cell_uninfected_2.png"
]

# --- PAGE CONFIG ---
st.set_page_config(page_title="MalariaScope · Vertical Space", layout="centered")

# ─── THE SPACESHIP UI (CSS - ปรับปรุงสไตล์อวกาศแนวตั้งของมึง) ──────────────────────────────
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
    font-size: 30px !important;
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

/* เลือก Mode สไตล์มินิมอล */
div[data-testid="stSelectbox"] {
    width: 100% !important;
    font-family: 'Orbitron', sans-serif !important;
}

div[data-testid="stSelectbox"] label {
    font-family: 'Orbitron' !important;
    color: #4da3ff !important;
    font-size: 0.8rem !important;
    letter-spacing: 1px;
}

/* --- ปุ่ม Action รันวิเคราะห์สไตล์อัจฉริยะ --- */
div.stButton {
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
    margin-top: 1.5rem !important;
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
    border: 1px solid rgba(77, 163, 255, 0.15);
    padding: 5px;
    background: rgba(0,0,0,0.5);
    width: 100%;
    margin-top: 15px;
}

/* ผลลัพธ์ (Result) แบบแนวตั้ง */
.result-display {
    background: rgba(255, 255, 255, 0.03);
    padding: 20px;
    text-align: center !important;
    border: 1px solid rgba(77, 163, 255, 0.2);
    margin-top: 20px;
    width: 100%;
}

.result-display * {
    text-align: center !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* MOBILE RESPONSIVE */
@media (max-width: 768px) {
    .hero-title {
        font-size: 22px !important; 
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
    <p style="font-family:Rajdhani; color:#4da3ff; letter-spacing:2px; margin-top:10px;">with 2-Class Optimization</p>
</div>
""", unsafe_allow_html=True)

# ─── MODEL LOADING & AUTO-FIX (ตัวซ่อมบั๊กเวอร์ชั่น Keras ของมึงสลัด) ─────────────────────
@st.cache_resource
def load_my_model():
    if not os.path.exists(FIXED_MODEL_PATH):
        if os.path.exists(MODEL_PATH):
            with zipfile.ZipFile(MODEL_PATH, 'r') as yin:
                with zipfile.ZipFile(FIXED_MODEL_PATH, 'w') as yout:
                    for item in yin.infolist():
                        data = yin.read(item.filename)
                        if item.filename == 'config.json':
                            config_json = json.loads(data.decode('utf-8'))
                            def remove_quantization(obj):
                                if isinstance(obj, dict):
                                    if 'quantization_config' in obj: del obj['quantization_config']
                                    for k, v in obj.items(): remove_quantization(v)
                                elif isinstance(obj, list):
                                    for i in obj: remove_quantization(i)
                            remove_quantization(config_json)
                            data = json.dumps(config_json).encode('utf-8')
                        yout.writestr(item, data)
        else:
            return None
    return tf.keras.models.load_model(FIXED_MODEL_PATH, compile=False)

try:
    model = load_my_model()
except Exception as e:
    st.error(f"🚀 SYSTEM ERROR: MODEL_NOT_FOUND_OR_FAILED")
    st.stop()

# ─── DATA SELECTION SECTION (ดึงจาก GitHub เมนูเดียวจบ ไม่ต้องมีปุ่มอัปโหลด) ───────────
choice = st.selectbox("CHOOSE DATASET (FROM GITHUB):", SAMPLE_IMAGES, label_visibility="visible")

img = None
if choice != "CHOOSE_DATASET...":
    img_url = GITHUB_RAW_URL + choice
    try:
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"❌ FETCH_ERROR: ไม่สามารถดึงรูปภาพจาก GitHub ได้")

# ─── SCANNING & RESULTS (งัดผลลัพธ์ออก 2 คลาส: ติดเชื้อ / ไม่ติดเชื้อ) ──────────────────
if img:
    st.markdown('<div class="img-container">', unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("START ANALYTICS"):
        # Preprocess ปรับขนาดรูปเข้าตัวโมเดลมึง
        img_p = img.resize(IMG_SIZE)
        img_arr = np.array(img_p)
        img_arr = np.expand_dims(img_arr, axis=0) # มิติ [1, 224, 224, 3] (มีเลเยอร์หาร 127.5 ในตัวแล้ว ข้ามไปได้เลยสลัด)
        
        with st.spinner("PROCESSING SYSTEM SCAN..."):
            predictions = model.predict(img_arr)[0]
        
        # ดึง 2 คลาสหลักมางัดกัน (Class 0: ติดเชื้อ, Class 1: ไม่ติดเชื้อ)
        prob_infected = predictions[0]
        prob_uninfected = predictions[1]
        
        # คำนวณสีและสถานะแยกขั้วความหล่อ
        if prob_infected > prob_uninfected:
            status = "INFECTED_DETECTED"
            color = "#ff3d6b"  # สีแดงสลัด ผงะเจอเชื้อ
            conf = prob_infected
        else:
            status = "NORMAL_CELL"
            color = "#00d2b4"  # สีเขียวมิ้นต์อวกาศ ปลอดภัย
            conf = prob_uninfected
        
        # พ่นการแสดงผลแนวตั้งครอบด้วยสไตล์ยานอวกาศตามแบบมึงเป๊ะ
        st.markdown(f"""
        <div class="result-display" style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
            <p style="font-family:Rajdhani; color:#8892b0; margin:0; font-size:0.8rem; text-transform:uppercase; width:100%;">SCAN RESULT</p>
            <h2 style="font-family:Orbitron; color:{color}; margin: 5px 0; letter-spacing:1px; font-size: 1.5rem; width:100%;">{status}</h2>
            <div style="margin: 10px auto; height: 1px; background: rgba(77,163,255,0.2); width: 100%;"></div>
            <p style="font-family:Rajdhani; color:#8892b0; margin:0; font-size:0.8rem; text-transform:uppercase; width:100%;">CONFIDENCE LEVEL</p>
            <h1 style="font-family:Orbitron; font-size:1.8rem; margin:0; color:#ffffff; width:100%;">{conf*100:.2f}%</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # พลอตกราฟเปรียบเทียบข้อมูล 2 คลาสแบบสตรีมลิตไว้ด้านล่างเท่ๆ 
        st.write("")
        chart_labels = {
            "ติดเชื้อ (Infected)": float(prob_infected),
            "ไม่ติดเชื้อ (Uninfected)": float(prob_uninfected)
        }
        st.bar_chart(chart_labels)
