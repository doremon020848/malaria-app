import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
MODEL_PATH = "best_model (6).keras"  # ชื่อไฟล์โมเดลของมึง
IMG_SIZE = (224, 224)
SAMPLE_DIR = "samples"

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
</style>
""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">Lightweight Image classification for<br>Malaria detection<br>using mobilenetv2</h1>
    <p class="hero-subtitle">3-Class Fixed Robust Prediction</p>
</div>
""", unsafe_allow_html=True)

# ─── MODEL LOADING ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    model = load_my_model()
except Exception as e:
    st.error(f"🚀 SYSTEM ERROR: MODEL_NOT_FOUND ({e})")
    st.stop()

# ─── SIDEBAR / CONTROLS (เพิ่มมาให้มึงปรับจูนและตรวจบั๊กง่ายขึ้น) ───
st.sidebar.header("🛠️ ตู้วิเคราะห์โมเดล")
# ให้มึงเลื่อนปรับ Threshold ได้เองบนเว็บ ถ้าตั้ง 0.00 คือปิดระบบดัก
CONFIDENCE_THRESHOLD = st.sidebar.slider("Confidence Threshold (เกณฑ์ดักภาพมั่ว)", 0.0, 1.0, 0.50, step=0.05)
show_debug = st.sidebar.checkbox("เปิด Debug Mode ดูค่าหลังบ้าน", value=True)

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
    st.image(img, use_container_width=True)
    
    if st.button("START ANALYTICS"):
        # Preprocess ภาพ
        img_p = img.resize(IMG_SIZE)
        img_arr = image.img_to_array(img_p)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = preprocess_input(img_arr)
        
        with st.spinner("PROCESSING..."):
            
            # 1. รันทำนายค่าจากโมเดล
            raw_predictions = model.predict(img_arr)
            
            # 🔥 บังคับแปลงค่าเอาต์พุตให้เป็น Softmax เผื่อเลเยอร์สุดท้ายของมึงไม่มี
            # สิ่งนี้จะทำให้ค่าความมั่นใจทั้ง 3 คลาสรวมกันได้ 1.0 (100%) เสมอ
            predictions = tf.nn.softmax(raw_predictions).numpy()
            
            # ลำดับคลาส (ถ้ามึงสลับโฟลเดอร์ตอนเทรน มาเปลี่ยนลำดับตรงนี้ได้)
            class_names = ['malaria', 'normal', 'others'] 
            
            max_confidence = float(np.max(predictions[0]))
            predicted_class_index = np.argmax(predictions[0])
            final_class = class_names[predicted_class_index]
            
            # 2. แสดงผล Debug Mode ฝั่งซ้ายมือ (เอาไว้ส่องดูเวลาโมเดลเอ๋อ)
            if show_debug:
                st.sidebar.subheader("📊 ค่าสถิติหลังบ้าน")
                st.sidebar.write(f"ค่าดิบจากโมเดล (Raw): {raw_predictions[0]}")
                st.sidebar.write(f"ค่าหลังแปลง Softmax: {predictions[0]}")
                st.sidebar.write(f"โมเดลเลือกคลาส: {final_class} (ดัชนี: {predicted_class_index})")
                st.sidebar.write(f"ความมั่นใจสูงสุด: {max_confidence*100:.2f}%")
            
            # 3. เช็กเงื่อนไขแสดงผลตามเกณฑ์ Threshold
            if max_confidence < CONFIDENCE_THRESHOLD:
                status = "NOT BLOOD CELL (LOW CONFIDENCE)"
                conf = max_confidence
                color = "#ff9f43"  # สีส้ม Warning
            else:
                conf = max_confidence
                if final_class == 'normal':
                    status = "NORMAL CELL"
                    color = "#00d2b4"  # สีเขียว
                elif final_class == 'malaria':
                    status = "INFECTED DETECTED"
                    color = "#ff3d6b"  # สีแดง
                else:  # คลาส others
                    status = "NOT BLOOD CELL (OTHERS)"
                    color = "#ff9f43"  # สีส้ม Warning
            
            # 4. พ่น UI แสดงผลลัพธ์
            st.markdown(f"""
                <div class="result-display">
                    <p style="font-family:'Inter', sans-serif; color:#8892b0; margin:0; font-size:0.8rem; text-transform:uppercase; font-weight:500;">SCAN RESULT</p>
                    <h2 style="font-family:'Inter', sans-serif; color:{color}; margin: 5px 0; letter-spacing:1px; font-size: 1.5rem; font-weight:700;">{status}</h2>
                    <p style="font-family:'Inter', sans-serif; color:#8892b0; margin:0; font-size:0.8rem; text-transform:uppercase; font-weight:500;">CONFIDENCE LEVEL</p>
                    <h1 style="font-family:'Inter', sans-serif; font-size:1.8rem; margin:0; color:#ffffff; font-weight:700;">{conf*100:.2f}%</h1>
                </div>
            """, unsafe_allow_html=True)
