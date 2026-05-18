import streamlit as st
import tensorflow as tf
import numpy as np
import os
import zipfile
import json
import requests
from PIL import Image
from io import BytesIO

# ----------------------------------------------------
# 1. ตั้งค่าลิงก์ GitHub Repository ของมึง (แก้ไขตรงนี้เลยสัส)
# ----------------------------------------------------
GITHUB_RAW_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/samples/"

# ใส่ชื่อไฟล์รูปภาพที่มีอยู่ในโฟลเดอร์ samples บน GitHub ของมึง
SAMPLE_IMAGES = [
    "เลือกรูปภาพตัวอย่าง...",
    "cell_infected_1.png",
    "cell_infected_2.png",
    "cell_uninfected_1.png",
    "cell_uninfected_2.png"
]

# ----------------------------------------------------
# 2. ฟังก์ชันโหลดและซ่อมแซมโมเดล Keras
# ----------------------------------------------------
@st.cache_resource
def load_malaria_model():
    model_path = 'best_model (6).keras'
    fixed_model_path = 'fixed_streamlit_model_v4.keras'
    
    if not os.path.exists(fixed_model_path):
        if os.path.exists(model_path):
            with zipfile.ZipFile(model_path, 'r') as yin:
                with zipfile.ZipFile(fixed_model_path, 'w') as yout:
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
    try:
        return tf.keras.models.load_model(fixed_model_path)
    except Exception as e:
        st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
        return None

model = load_malaria_model()

# ----------------------------------------------------
# 3. ออกแบบหน้าต่างเว็บอินเตอร์เฟซ
# ----------------------------------------------------
st.set_page_config(page_title="Malaria Detection (2 Classes)", page_icon="🔬", layout="centered")

st.title("🔬 ระบบวิเคราะห์เชื้อมาลาเรีย")
st.write("ไอ้ยี่สิบ! หน้าเว็บนี้ปรับเหลือแค่ **2 คลาส** เน้นๆ ชัดๆ: **ติดเชื้อ** กับ **ไม่ติดเชื้อ** เท่านั้นสัส")

# สร้างแท็บ 2 ช่องทาง
tab1, tab2 = st.tabs(["📂 ดึงรูปจาก GitHub (Samples)", "📤 อัปโหลดรูปภาพใหม่"])

selected_image = None

# --- แท็บที่ 1: ดึงรูปจาก GitHub ---
with tab1:
    selected_file_name = st.selectbox("เลือกไฟล์รูปภาพจาก GitHub ของมึง:", SAMPLE_IMAGES)
    if selected_file_name != "เลือกรูปภาพตัวอย่าง...":
        img_url = GITHUB_RAW_URL + selected_file_name
        try:
            response = requests.get(img_url)
            selected_image = Image.open(BytesIO(response.content))
            st.info(f"🔗 ดึงรูปสำเร็จจาก: `{img_url}`")
        except Exception as e:
            st.error(f"❌ ดึงรูปจาก GitHub ไม่ผ่านสลัด เช็ค URL ดีๆ: {e}")

# --- แท็บที่ 2: อัปโหลดรูปภาพเอง ---
with tab2:
    uploaded_file = st.file_uploader("📤 โยนรูปภาพเซลล์เม็ดเลือดเข้ามา...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        selected_image = Image.open(uploaded_file)

# ----------------------------------------------------
# 4. ส่วนประมวลผลคำนวณและพยากรณ์ผลลัพธ์ (งัดกันแค่ 2 คลาสตามสั่ง)
# ----------------------------------------------------
if selected_image is not None:
    st.write("---")
    st.image(selected_image, caption='📷 รูปภาพที่ประมวลผล', use_container_width=True)
    
    if model is None:
        st.error("❌ ไม่พบไฟล์โมเดล 'best_model (6).keras' ในโฟลเดอร์นี้เว้ย!")
    else:
        with st.spinner('⏳ AI กำลังสแกนหาเชื้อแป๊บ...'):
            # 1. ทำพรีโพรเซสเซสซิ่ง
            img_resized = selected_image.convert("RGB").resize((224, 224))
            img_array = np.array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            
            # 2. สั่งโมเดลทำนายผลลัพธ์
            predictions = model.predict(img_array)[0]
            
            # ดึงเฉพาะคลาส 0 และ คลาส 1 มางัดกันตรงๆ (คลาส 2 ช่างหัวมัน ไม่เอามาแสดง)
            prob_infected = predictions[0]
            prob_uninfected = predictions[1]
            
            st.subheader("📊 ผลการวิเคราะห์จากระบบ")
            
            # ตัดสินผลลัพธ์จาก 2 คลาสนี้อันไหนคะแนนสูงกว่าชนะ
            if prob_infected > prob_uninfected:
                st.error(f"🚨 ผลการตรวจวิเคราะห์: **ติดเชื้อมาลาเรีย (Infected / Parasitized)**")
                st.metric(label="ค่าความมั่นใจ (Confidence)", value=f"{prob_infected*100:.2f}%")
            else:
                st.success(f"✅ ผลการตรวจวิเคราะห์: **ไม่ติดเชื้อ (Uninfected / Normal)**")
                st.metric(label="ค่าความมั่นใจ (Confidence)", value=f"{prob_uninfected*100:.2f}%")

            # 4. แสดงกราฟแท่งแบบ 2 คลาสเพียวๆ ให้เห็นชัดเจน
            st.write("---")
            st.subheader("📈 กราฟแสดงสถิติเปรียบเทียบ 2 คลาส")
            chart_labels = {
                "ติดเชื้อ (Infected)": float(prob_infected),
                "ไม่ติดเชื้อ (Uninfected)": float(prob_uninfected)
            }
            st.bar_chart(chart_labels)
