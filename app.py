import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
MODEL_PATH = "best_model.keras" # ชื่อไฟล์โมเดลที่มึงเทรนเสร็จ
IMG_SIZE = (224, 224)

# --- UI SETTINGS ---
st.set_page_config(page_title="Malaria Detection App", layout="centered")
st.title("🔬 Malaria Cell Classification")
st.write("อัปโหลดรูปภาพเซลล์เม็ดเลือดแดง เพื่อตรวจสอบการติดเชื้อมาลาเรีย")

# --- LOAD MODEL ---
@st.cache_resource
def load_my_model():
    # โหลดโมเดลที่มึงเทรนจากไฟล์ .keras
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_my_model()
    st.success("โหลดโมเดลสำเร็จ!")
except Exception as e:
    st.error(f"ยังหาไฟล์ {MODEL_PATH} ไม่เจอ มึงอย่าลืมเอาไฟล์โมเดลมาวางไว้ในโฟลเดอร์เดียวกันนะ!")
    st.stop()

# --- UPLOAD FILE ---
uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพเซลล์ (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. แสดงรูปภาพที่อัปโหลด
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="รูปที่อัปโหลด", use_column_width=True)
    
    # 2. Preprocessing (ทำเหมือนตอนเทรนเป๊ะๆ)
    img_resized = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) # เพิ่มมิติ batch
    img_array = preprocess_input(img_array)      # Normalize ตาม MobileNetV2

    # 3. Prediction
    if st.button("เริ่มวิเคราะห์ผล"):
        with st.spinner('กำลังประมวลผล...'):
            prediction = model.predict(img_array)[0][0]
            
            # เนื่องจากมึงใช้ Activation เป็น Sigmoid (Dense(1))
            # ค่าที่ได้จะอยู่ระหว่าง 0 ถึง 1
            # โดยปกติ 0.5 คือจุดแบ่ง (Threshold)
            
            st.divider()
            st.subheader("📊 ผลการวิเคราะห์:")
            
            # ปรับแต่งการแสดงผลตาม Class ของมึง (ปกติ 0=Infected, 1=Uninfected หรือสลับกันตามลำดับโฟลเดอร์)
            # ในที่นี้สมมติว่าค่าสูงเข้าใกล้ 1 คือ Uninfected (ปกติ) และเข้าใกล้ 0 คือ Infected (ติดเชื้อ)
            # มึงต้องเช็ค train_gen.class_indices ในโค้ดเทรนอีกทีนะ
            
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            if prediction > 0.5:
                label = "Uninfected (ปกติ)"
                st.balloons()
                st.success(f"ผลลัพธ์: **{label}**")
            else:
                label = "Infected (ติดเชื้อมาลาเรีย)"
                st.warning(f"ผลลัพธ์: **{label}**")
                
            st.write(f"ความเชื่อมั่น (Confidence): **{confidence*100:.2f}%**")
            st.progress(float(confidence))

# --- FOOTER ---
st.divider()
st.caption("พัฒนาโดยไอ้ยี่สิบ | Data Science CMU")
