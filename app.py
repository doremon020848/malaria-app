import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image

# --- CONFIGURATION & THEME ---
st.set_page_config(page_title="Malaria AI Detector", layout="wide", page_icon="🔬")

# ปรับแต่ง CSS ให้เป็นธีม แดง-ดำ (Red-Black Neon)
st.markdown("""
    <style>
    /* พื้นหลังหลักและสีตัวอักษร */
    .stApp {
        background-color: #0b0c10;
        color: #eeeeee;
    }
    /* หัวข้อสีแดง Neon */
    h1, h2, h3 {
        color: #ff0000 !important;
        text-shadow: 0 0 10px #ff0000, 0 0 20px #800000;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    /* การตกแต่ง Sidebar และกล่อง Upload */
    .stFileUploader {
        border: 2px dashed #ff0000;
        background-color: #1f2833;
        border-radius: 15px;
    }
    /* ปรับแต่งปุ่ม */
    .stButton>button {
        width: 100%;
        background-color: #8b0000;
        color: white;
        border: 2px solid #ff0000;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff0000;
        box-shadow: 0 0 20px #ff0000;
        color: white;
    }
    /* ผลลัพธ์สีแดง-เขียว */
    .stAlert {
        border-radius: 10px;
        border: 1px solid #ff0000;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_malaria_model():
    # ตรวจสอบว่าไฟล์ชื่อตรงกับที่ save ไว้ (malaria_mobilenetv2_model.keras)
    return tf.keras.models.load_model('malaria_mobilenetv2_model.keras')

try:
    model = load_malaria_model()
    # กำหนดคลาสตามที่เทรนไว้ (0: malaria, 1: normal จากโค้ดต้นฉบับ)
    class_names = ['Malaria Detected', 'Normal Cell'] 
except Exception as e:
    st.error(f"ไม่พบไฟล์โมเดล กรุณาตรวจสอบชื่อไฟล์: {e}")

# --- APP LAYOUT ---
st.title("🔬 Malaria Cell Analysis System")
st.write("AI-Powered Diagnostic Assistance for Medical Research")
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Image Submission")
    uploaded_file = st.file_uploader("Upload blood cell image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        # ตกแต่งกรอบรูปในคอลัมน์ซ้าย
        st.image(img, caption='Submitted Specimen', use_container_width=True)

with col2:
    st.subheader("🧪 Diagnostic Output")
    if uploaded_file:
        # --- PREPROCESSING ---
        # ปรับขนาดรูปให้ตรงกับ input ของ MobileNetV2 (224x224)
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # --- PREDICTION ---
        with st.spinner('AI analyzing morphology...'):
            preds = model.predict(img_array)
            idx = np.argmax(preds[0])
            label = class_names[idx]
            confidence = preds[0][idx] * 100

        # --- DISPLAY RESULTS ---
        if "Malaria" in label:
            st.error(f"## {label}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            st.progress(int(confidence))
            st.warning("🚨 Morphology suggests parasitic infection. Further clinical validation required.")
        else:
            st.success(f"## {label}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            st.progress(int(confidence))
            st.info("✅ Specimen appears within normal parameters.")

    else:
        st.info("Awaiting specimen upload for automated analysis.")

# --- FOOTER ---
st.markdown("---")
st.caption("Developed for: Malaria Classification Competition | Model: MobileNetV2 | Tier: Pro AI")
