import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
MODEL_PATH = "best_model_lite.h5"  # ใช้ชื่อไฟล์ใหม่ที่กุรีดไขมันให้
IMG_SIZE = (224, 224)
SAMPLE_DIR = "samples"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="MalariaScope · CMU",
    page_icon="🧬",
    layout="centered",
)

# ─── MASTER CSS (กูรวมตัวตึงไว้ให้เหมือนเดิม) ──────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Outfit:wght@300;400;500&display=swap');
:root { --ink: #04080f; --cyan: #00d2b4; --blue: #0a84ff; --red: #ff3d6b; --muted: #4d7a99; --txt: #d8eeff; --border2: rgba(0,210,180,0.28); --r: 18px; }
html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; background-color: var(--ink) !important; color: var(--txt) !important; }
.grid-bg { position: fixed; inset: 0; z-index: 0; background-image: linear-gradient(rgba(0,210,180,.04) 1px, transparent 1px), linear-gradient(90deg, rgba(0,210,180,.04) 1px, transparent 1px); background-size: 48px 48px; pointer-events: none; }
.hero-title { font-family: 'Syne', sans-serif; font-size: 3rem; font-weight: 800; text-align: center; background: linear-gradient(160deg, #ffffff, var(--cyan), var(--blue)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.section-label { font-family: 'Space Mono', monospace; font-size: .65rem; letter-spacing: .18em; color: var(--muted); text-transform: uppercase; margin: 1.5rem 0 1rem; display: flex; align-items: center; gap: .5rem; }
.section-label::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, var(--border2), transparent); }
.result-box { border-radius: var(--r); padding: 2rem; text-align: center; border: 1px solid var(--border2); margin-top: 1.5rem; }
.conf-pct { font-family: 'Space Mono', monospace; font-size: 2.5rem; font-weight: 700; }
[data-testid="stButton"] > button { width: 100% !important; background: linear-gradient(135deg, var(--cyan), var(--blue)) !important; color: #040c18 !important; font-weight: 700 !important; border-radius: 12px !important; }
</style>
<div class="grid-bg"></div>
""", unsafe_allow_html=True)

st.markdown('<h1 class="hero-title">MalariaScope</h1>', unsafe_allow_html=True)

# ─── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    model = load_my_model()
except Exception:
    st.error(f"⚠️ ไม่พบไฟล์ `{MODEL_PATH}` อัปโหลดขึ้น GitHub ด้วยนะไอ้ยี่สิบ!")
    st.stop()

# ─── INPUT SELECTION ──────────────────────────────────────────────────────────
st.markdown('<div class="section-label">⚙️ การเลือกนำเข้าข้อมูล</div>', unsafe_allow_html=True)
input_mode = st.radio("เลือกวิธีนำเข้ารูปภาพ:", ["เลือกจากคลังตัวอย่าง (Samples)", "อัปโหลดรูปภาพใหม่เอง (Upload)"], horizontal=True)

img = None
filename = ""

if input_mode == "เลือกจากคลังตัวอย่าง (Samples)":
    if os.path.exists(SAMPLE_DIR):
        sample_images = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if sample_images:
            selected_file = st.selectbox("เลือกภาพจาก samples:", sample_images)
            img_path = os.path.join(SAMPLE_DIR, selected_file)
            img = Image.open(img_path).convert("RGB")
            filename = selected_file
        else:
            st.warning("ไม่มีรูปในโฟลเดอร์ samples ว่ะ")
    else:
        st.error(f"ไม่พบโฟลเดอร์ `{SAMPLE_DIR}`")
else:
    uploaded_file = st.file_uploader("อัปโหลดภาพเซลล์เม็ดเลือด (JPG/PNG):", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        filename = uploaded_file.name

# ─── PREVIEW & ANALYSIS ───────────────────────────────────────────────────────
if img:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img, caption=f"รูปที่เลือก: {filename}", use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-label">📊 ผลการวิเคราะห์</div>', unsafe_allow_html=True)
        
        # Preprocess
        img_input = img.resize(IMG_SIZE)
        img_array = image.img_to_array(img_input)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        if st.button("🔬 ANALYSE CELL"):
            with st.spinner("AI กำลังวินิจฉัย..."):
                prediction = float(model.predict(img_array)[0][0])
            
            is_safe = prediction > 0.5
            confidence = prediction if is_safe else 1 - prediction
            
            color = "#00d2b4" if is_safe else "#ff3d6b"
            status = "✅ ไม่ติดเชื้อ (Normal)" if is_safe else "⚠️ ตรวจพบเชื้อ (Infected)"
            
            st.markdown(f"""
            <div class="result-box">
                <div style="font-size: 1.2rem; color: {color}; font-weight: bold;">{status}</div>
                <div class="conf-pct" style="color: {color};">{confidence*100:.2f}%</div>
                <div style="font-size: 0.7rem; color: #4d7a99;">CONFIDENCE SCORE</div>
            </div>
            """, unsafe_allow_html=True)
            if is_safe: st.balloons()

st.markdown("""<div style="text-align: center; margin-top: 3rem; color: #4d7a99; font-size: 0.7rem;">🧬 MalariaScope · CMU DATA SCIENCE · 2026</div>""", unsafe_allow_html=True)
