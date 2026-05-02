import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import os

# --- CONFIGURATION & THEME ---
st.set_page_config(page_title="Malaria AI Detector", layout="wide", page_icon="🔬")

# ตกแต่งธีม Red-Black ตามที่คุณต้องการ
st.markdown("""
    <style>
    .stApp { background-color: #0b0c10; color: #eeeeee; }
    h1, h2, h3 { 
        color: #ff0000 !important; 
        text-shadow: 0 0 10px #ff0000; 
        font-family: 'Segoe UI', sans-serif;
    }
    .stFileUploader { border: 2px dashed #ff0000; background-color: #1f2833; border-radius: 15px; }
    .stButton>button { 
        width: 100%; background-color: #8b0000; color: white; 
        border: 2px solid #ff0000; border-radius: 8px; font-weight: bold; 
    }
    .stButton>button:hover { background-color: #ff0000; box-shadow: 0 0 20px #ff0000; }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_malaria_model():
    # ไฟล์ต้องชื่อตรงเป๊ะกับที่อัปโหลดขึ้น GitHub
    model_path = 'malaria_mobilenetv2_model.keras'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = None
class_names = ['Malaria Detected', 'Normal Cell']

try:
    model = load_malaria_model()
except Exception as e:
    st.error(f"❌ โหลดโมเดลไม่ได้: {e}")

# --- APP LAYOUT ---
st.title("🔬 Malaria Cell Analysis System")
st.write("AI-Powered Diagnostic Assistance for Medical Research")
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Image Submission")
    uploaded_file = st.file_uploader("Upload blood cell image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption='Submitted Specimen', use_container_width=True)

with col2:
    st.subheader("🧪 Diagnostic Output")
    if uploaded_file:
        if model is not None:
            # Preprocessing ให้เข้ากับ MobileNetV2[cite: 1]
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            try:
                with st.spinner('AI analyzing morphology...'):
                    preds = model.predict(img_array)
                    idx = np.argmax(preds[0])
                    label = class_names[idx]
                    confidence = preds[0][idx] * 100

                # แสดงผลลัพธ์
                if "Malaria" in label:
                    st.error(f"## {label}")
                    st.write(f"**Confidence:** {confidence:.2f}%")
                    st.progress(int(confidence))
                else:
                    st.success(f"## {label}")
                    st.write(f"**Confidence:** {confidence:.2f}%")
                    st.progress(int(confidence))
            except Exception as ex:
                st.error(f"วิเคราะห์ล้มเหลว: {ex}")
        else:
            st.warning("⚠️ โมเดลไม่พร้อมใช้งาน ตรวจสอบไฟล์โมเดลบน GitHub")
    else:
        st.info("Awaiting specimen upload...")
