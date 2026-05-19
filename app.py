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
}

/* ผลลัพธ์ (Result) */
.result-display {
    background: rgba(255, 255, 255, 0.03);
    padding: 20px;
    text-align: center !important; /* บังคับกลางตรงนี้ */
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
    <p style="font-family:Rajdhani; color:#4da3ff; letter-spacing:2px; margin-top
