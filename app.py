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
st.set_page_config(page_title="Malaria Detection System", layout="centered")

# ─── MEDICAL CLEAN UI (CSS) ──────────────────────────────
st.markdown("""
<style>
/* พื้นหลังสีฟ้าเทาอ่อนๆ ให้ฟีลสะอาดแบบโรงพยาบาล */
.stApp {
    background-color: #f4f7f9;
}

#MainMenu, footer, header { visibility: hidden; }

/* หัวข้อโปรเจกต์ */
.header-container {
    text-align: center;
    padding: 30px 0;
}
.main-title {
    font-family: 'Helvetica Neue', sans-serif;
    color: #2c3e50;
    font-size: 1.6rem;
    font-weight: 700;
    line-height: 1.4;
}

/* ส่วนของ Selection Mode */
.mode-container {
    display: flex;
    gap: 20px;
    margin-bottom: 10px;
}

/* กล่องขาวสำหรับเนื้อหา */
.content-card {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #e1e8ed;
    box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    margin-bottom: 20px;
}

/* ปุ่ม Analysis สไตล์โปร */
div.stButton > button {
    width: 100% !important;
    background: linear-gradient(90deg, #4da3ff, #6a82fb) !important;
    color: white !important;
    border: none !important;
    height: 3.5rem !important;
    border-radius: 8px !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    margin-top: 10px;
}

/* Result Box */
.result-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    border-top: 4px solid #4da3ff;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <h1 class="main-title">LIGHTWEIGHT IMAGE CLASSIFICATION FOR<br>MALARIA DETECTION USING MOBILENETV2</h1>
</div>
""", unsafe_allow_html=True)

# ─── DATA INPUT SECTION ──────────────────────────────────────────────────────
col_input, col_dataset = st.columns([1, 1])

with col_input:
    st.markdown('**SELECTION MODE**')
    # ปรับเป็น Radio สไตล์มินิมอลแทน Checkbox เพื่อให้เลือกได้อย่างใดอย่างหนึ่งตามปกติของ UI
    mode = st.radio("", ["Samples", "Upload"], horizontal=True, label_visibility="collapsed")

with col_dataset:
    st.markdown('**CHOOSE DATASET**')
    if mode == "Samples":
        if os.path.exists(SAMPLE_DIR):
            files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            choice = st.selectbox("", files, label_visibility="collapsed")
            img_path = os.path.join(SAMPLE_DIR, choice) if choice else None
            if img_path: img = Image.open(img_path).convert("RGB")
        else:
            st.error("Sample folder not found.")
            img = None
    else:
        up = st.file_uploader("", type=["jpg", "png"], label_visibility="collapsed")
        img = Image.open(up).convert("RGB") if up else None

# ─── IMAGE DISPLAY & ANALYSIS ────────────────────────────────────────────────
if img:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.image(img, use_container_width=True, caption="Selected Red Blood Cell Microscope Image")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("START MALARIA DIAGNOSIS ANALYSIS"):
        # --- (Place your Model Prediction Logic here) ---
        # สมมติผลลัพธ์
        status = "POSITIVE (Malaria Infected)"
        confidence = 98.45
        color = "#e74c3c" # สีแดงสำหรับติดเชื้อ
        
        # ─── RESULTS DISPLAY (เหลือ 2 คอลัมน์ตามสั่ง) ──────────────────────────
        st.markdown('<br>', unsafe_allow_html=True)
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown(f"""
            <div class="result-card">
                <p style="color:#7f8c8d; font-size:0.9rem; margin-bottom:5px;">DIAGNOSIS RESULT</p>
                <h3 style="color:{color}; margin:0;">{status}</h3>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            st.markdown(f"""
            <div class="result-card">
                <p style="color:#7f8c8d; font-size:0.9rem; margin-bottom:5px;">CONFIDENCE LEVEL</p>
                <h3 style="color:#2c3e50; margin:0;">{confidence}%</h3>
            </div>
            """, unsafe_allow_html=True)

        # ─── GRAD-CAM SECTION ────────────────────────────────────────────────
        st.markdown('<br>**GRAD-CAM VISUALIZATION**', unsafe_allow_html=True)
        st.markdown("""
        <div class="content-card" style="text-align:center; min-height:200px; color:#bdc3c7;">
            [ Heatmap Visualization ]
        </div>
        """, unsafe_allow_html=True)

# ลบส่วน Contact us ออกเรียบร้อยตามบรีฟ
