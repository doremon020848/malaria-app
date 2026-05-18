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
    fixed_model_path = 'fixed_streamlit_model.keras'
    
    # แก้ไขบั๊กโครงสร้าง Keras เวอร์ชั่นไม่ตรงกันให้อัตโนมัติสัส
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

# สั่งให้คลาสโหลดโมเดลสแตนด์บายรอไว้เลย
model = load_malaria_model()

# ----------------------------------------------------
# 2. ออกแบบหน้าต่างเว็บอินเตอร์เฟซ
# ----------------------------------------------------
st.set_page_config(page_title="Malaria & Dengue Detection", page_icon="🔬", layout="centered")

st.title("🔬 ระบบวิเคราะห์ภาพถ่ายเชื้อมาลาเรียและไข้เลือดออก")
st.write("ไอ้ยี่สิบ! มึงลองอัปโหลดรูปภาพผลเลือด หรือรูปเซลล์ เพื่อให้ AI ทำนายผลลัพธ์แยกคลาสได้เลย")

# ตั้งชื่อคลาสแสดงผลให้สอดคล้องกับโมเดล 3 คลาสของมึง
CLASS_NAMES = [
    "เชื้อมาลาเรีย (Malaria)", 
    "ไข้เลือดออก / เซลล์ปกติ (Dengue / Normal)", 
    "ไม่เกี่ยวข้อง / ไม่ใช่รูปตรวจเชื้อ (Unrelated)"
]

# ช่องอัปโหลดไฟล์รูปภาพบนหน้าเว็บ
uploaded_file = st.file_uploader("โยนรูปภาพเข้ามารันตรงนี้สลัด...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # เปิดรูปภาพขึ้นมาโชว์บนหน้าเว็บ
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption='📷 รูปภาพที่มึงอัปโหลดเข้ามา', use_container_width=True)
    
    if model is None:
        st.error("❌ ไม่พบไฟล์โมเดล 'best_model (6).keras' ในโฟลเดอร์เดียวกับโค้ดเว็บนี้เว้ย!")
    else:
        with st.spinner('⏳ AI กำลังขยี้รูปภาพประมวลผลคำนวณกราฟแป๊บ...'):
            # 1. ทำพรีโพรเซสเซสซิ่ง ปรับขนาดรูปเป็น 224x224 ตามที่โมเดลระบุไว้
            img_resized = image_display.convert("RGB").resize((224, 224))
            img_array = np.array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) # ขยายมิติเป็น (1, 224, 224, 3)
            
            # 2. สั่งโมเดลทำนายผลลัพธ์ดึงเอาค่ามาใช้งาน
            predictions = model.predict(img_array)[0]
            predicted_class_idx = np.argmax(predictions)
            confidence = predictions[predicted_class_idx]
            
            st.write("---")
            st.subheader("📊 ผลการวิเคราะห์จากโมเดล")
            
            # 3. ตั้งเงื่อนไขแยกหมวดหมู่ตามที่มึงสั่งมา
            # ถ้าโมเดลชี้ไปที่คลาส 2 หรือค่าความน่าจะเป็นต่ำกว่า 50% ให้ตีเป็นรูปคนอื่น/ไม่เกี่ยวข้องทันที
            if predicted_class_idx == 2 or confidence < 0.5:
                st.warning("⚠️ ผลลัพธ์: ไม่เกี่ยวข้อง / ไม่ใช่รูปภาพผลตรวจเชื้อมาลาเรียหรือไข้เลือดออก")
                st.info(f"💡 ระบบคัดออกไปหมวดรูปภาพอื่น (ความมั่นใจสูงสุดเพียง: {confidence*100:.2f}%)")
            else:
                # ถ้าทายถูกกลุ่มคลาสมาลาเรีย หรือไข้เลือดออกแดงปกติ
                st.success(f"🎯 ผลการตรวจวิเคราะห์: **{CLASS_NAMES[predicted_class_idx]}**")
                st.metric(label="ค่าความมั่นใจสูงสุด (Confidence)", value=f"{confidence*100:.2f}%")
                
            # 4. ดึงฟังก์ชันพลอตกราฟแท่งแสดงข้อมูล Output มาโชว์ด้านล่างแยกกัน
            st.write("---")
            st.subheader("📈 กราฟแสดงสถิติค่าน้ำหนักความน่าจะเป็นย้อนหลัง")
            chart_data = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
            st.bar_chart(chart_data) # พลอตกราฟแท่งสวยๆ ของ Streamlit ให้เลยข้ามฟังก์ชันสบายๆ
