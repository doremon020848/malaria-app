import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="Malaria AI Detector", layout="wide", page_icon="🔬")

# ตกแต่งธีม Red-Black
st.markdown("""
    <style>
    .stApp { background-color: #0b0c10; color: #eeeeee; }
    h1, h2, h3 { color: #ff0000 !important; text-shadow: 0 0 10px #ff0000; }
    .stFileUploader { border: 2px dashed #ff0000; background-color: #1f2833; }
    .sample-card { 
        border: 1px solid #444; padding: 10px; border-radius: 10px; 
        text-align: center; cursor: pointer; background: #1f2833;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_malaria_model():
    model_path = 'best_model.keras'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_malaria_model()
class_names = ['Malaria Detected', 'Normal Cell']

# --- SIDEBAR: SAMPLE IMAGES ---
st.sidebar.header("📁 Sample Specimens")
st.sidebar.write("เลือกรูปตัวอย่างเพื่อทดสอบระบบ")

# สร้างรายการรูปตัวอย่าง (มึงต้องมีไฟล์เหล่านี้ในโฟลเดอร์ samples นะไอ้สัส)
sample_folder = "samples"
samples = {
    "Malaria Example 1": f"{sample_folder}/malaria_1.png",
    "Normal Example 1": f"{sample_folder}/normal_1.png"
}

selected_sample = st.sidebar.selectbox("เลือกรูปตัวอย่าง", ["None"] + list(samples.keys()))

# --- MAIN APP ---
st.title("🔬 Malaria Cell Analysis System")
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Image Submission")
    
    # ส่วนอัปโหลดไฟล์จากเครื่อง (Local Upload)
    uploaded_file = st.file_uploader("Upload blood cell image from your device", type=["jpg", "jpeg", "png"])
    
    final_img = None

    # Logic การเลือกรูป: ถ้าอัปโหลดให้ใช้รูปอัปโหลด ถ้าเลือก sample ให้ใช้ sample
    if uploaded_file:
        final_img = Image.open(uploaded_file).convert('RGB')
        st.info("ใช้รูปภาพจากการอัปโหลด")
    elif selected_sample != "None":
        sample_path = samples[selected_sample]
        if os.path.exists(sample_path):
            final_img = Image.open(sample_path).convert('RGB')
            st.info(f"ใช้รูปตัวอย่าง: {selected_sample}")
        else:
            st.error(f"ไม่เจอไฟล์ตัวอย่างที่: {sample_path}")

    if final_img:
        st.image(final_img, caption='Specimen for Analysis', use_container_width=True)

with col2:
    st.subheader("🧪 Diagnostic Output")
    if final_img:
        if model is not None:
            try:
                with st.spinner('AI analyzing morphology...'):
                    # Preprocessing
                    img_resized = final_img.resize((224, 224))
                    img_array = image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)

                    # Predict
                    preds = model.predict(img_array)
                    idx = np.argmax(preds[0])
                    label = class_names[idx]
                    confidence = preds[0][idx] * 100

                    # Result UI
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
            st.warning("⚠️ โมเดลไม่พร้อมใช้งาน")
    else:
        st.info("Awaiting specimen upload or sample selection...")
