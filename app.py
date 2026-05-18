import streamlit as st
import tensorflow as tf
import numpy as np
import os
import zipfile
import json
from PIL import Image

# ----------------------------------------------------
# 1. ฟังก์ชันโหลดและซ่อมแซมโมเดล Keras แช่ไว้ในหน่วยความจำ
# ----------------------------------------------------
@st.cache_resource
def load_malaria_model():
    model_path = 'best_model (6).keras'
    fixed_model_path = 'fixed_streamlit_model_v2.keras'
    
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
# 2. ออกแบบหน้าต่างเว็บอินเตอร์เฟซ
# ----------------------------------------------------
st.set_page_config(page_title="Malaria Detection (ติดเชื้อ/ไม่ติดเชื้อ)", page_icon="🔬", layout="centered")

st.title("🔬 ระบบวิเคราะห์ภาพถ่ายเชื้อมาลาเรีย")
st.write("ไอ้ยี่สิบ! อัปโหลดรูปภาพผลเลือดจากกล้องจุลทรรศน์เพื่อเช็คว่า **ติดเชื้อ** หรือ **ไม่ติดเชื้อ**")

# ----------------------------------------------------
# ช่องอัปโหลดไฟล์รูปภาพบนหน้าเว็บ
# ----------------------------------------------------
uploaded_file = st.file_uploader("โยนรูปภาพเซลล์เม็ดเลือดเข้ามาตรงนี้เลยสัส...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # เปิดรูปภาพขึ้นมาโชว์บนหน้าเว็บ
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption='📷 รูปภาพที่มึงอัปโหลดเข้ามา', use_container_width=True)
    
    if model is None:
        st.error("❌ ไม่พบไฟล์โมเดล 'best_model (6).keras' ในโฟลเดอร์เดียวกับโค้ดเว็บนี้เว้ย!")
    else:
        with st.spinner('⏳ AI กำลังส่องกล้องวิเคราะห์รูปภาพแป๊บ...'):
            # 1. ปรับขนาดรูปภาพให้เข้ากับโมเดล (224x224)
            img_resized = image_display.convert("RGB").resize((224, 224))
            img_array = np.array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            
            # 2. สั่งโมเดลทำนายผลลัพธ์
            predictions = model.predict(img_array)[0]
            predicted_class_idx = np.argmax(predictions)
            confidence = predictions[predicted_class_idx]
            
            st.write("---")
            st.subheader("📊 ผลการวิเคราะห์จากระบบ")
            
            # 3. เช็คเงื่อนไขคัดกรองรูปภาพ "ไม่เกี่ยวข้อง" ออกไปก่อนตามที่มึงสั่งไว้
            # ถ้าโมเดลชี้ไปคลาส 2 หรือโมเดลทายไม่ขาด (ค่าความมั่นใจต่ำกว่า 60%) ดันออกไปหมวดไม่เกี่ยวข้องทันที
            if predicted_class_idx == 2 or confidence < 0.60:
                st.warning("⚠️ ผลการตรวจสอบ: ไม่เกี่ยวข้อง / ไม่ใช่รูปภาพเซลล์เม็ดเลือดที่ใช้ตรวจเชื้อ")
                st.info(f"💡 ระบบคัดออก: โมเดลมองว่าเป็นรูปภาพอื่นที่ไม่เกี่ยวข้อง (ความมั่นใจ {confidence*100:.2f}%)")
            
            else:
                # ถ้ามั่นใจว่าเป็นรูปเม็ดเลือดชัวร์ๆ ค่อยแยกคำตอบเป็น ติดเชื้อ กับ ไม่ติดเชื้อ
                if predicted_class_idx == 0:
                    st.error(f"🚨 ผลการตรวจวิเคราะห์: **ติดเชื้อมาลาเรีย (Infected / Parasitized)**")
                    st.metric(label="ค่าความมั่นใจ (Confidence)", value=f"{confidence*100:.2f}%")
                elif predicted_class_idx == 1:
                    st.success(f"✅ ผลการตรวจวิเคราะห์: **ไม่ติดเชื้อ (Uninfected / Normal)**")
                    st.metric(label="ค่าความมั่นใจ (Confidence)", value=f"{confidence*100:.2f}%")

            # 4. พลอตกราฟแท่งโชว์เปรียบเทียบค่าน้ำหนักความน่าจะเป็นย้อนหลัง
            st.write("---")
            st.subheader("📈 กราฟแสดงสถิติค่าน้ำหนักความน่าจะเป็น")
            chart_labels = {
                "ติดเชื้อ (Class 0)": float(predictions[0]),
                "ไม่ติดเชื้อ (Class 1)": float(predictions[1]),
                "รูปอื่น/ไม่เกี่ยวข้อง (Class 2)": float(predictions[2])
            }
            st.bar_chart(chart_labels)
