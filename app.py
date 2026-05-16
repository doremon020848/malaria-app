import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# ==========================================
# 1. CONFIG & SETTINGS (ตั้งค่าหน้าเว็บและโมเดล)
# ==========================================
st.set_page_config(
    page_title="Malaria Detection Research App",
    page_icon="🔬",
    layout="centered"
)

# 📌 [จุดสำคัญมาก] ปรับแต่งตรงนี้ให้ตรงกับโมเดลในงานวิจัยของมึง!
MODEL_PATH = "best_model (6).keras"

# 1️⃣ เช็คขนาดภาพตอนเทรน: ลองเปลี่ยนเป็น (100, 100), (128, 128), (150, 150) หรือ (224, 224) ตามที่มึงใช้
TARGET_IMAGE_SIZE = (100, 100)  

# 2️⃣ เช็คประเภทภาพตอนเทรน: True = ภาพสี (RGB) | False = ภาพขาวดำ (Grayscale)
IS_COLOR_IMAGE = True  

SAMPLE_DIR = "samples" # ชื่อโฟลเดอร์รูปตัวอย่างบน GitHub

# ==========================================
# 2. LOAD MODEL (โหลดโมเดล)
# ==========================================
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_my_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ โหลดโมเดลไม่สำเร็จ! เช็คว่าไฟล์ '{MODEL_PATH}' อยู่ที่เดียวกับโค้ดนี้ไหม")
    st.info(f"รายละเอียด Error: {e}")

# ==========================================
# 3. HELPER FUNCTIONS (ฟังก์ชันประมวลผลภาพ)
# ==========================================
def preprocess_image(image, target_size=TARGET_IMAGE_SIZE, is_color=IS_COLOR_IMAGE):
    """ แปลงรูปภาพให้อยู่ในมิติ (Shape) ที่โมเดลของมึงยอมรับ """
    # 1. ตรวจสอบและแปลงประเภทสี (RGB หรือ Grayscale)
    if is_color:
        img = image.convert('RGB')
    else:
        img = image.convert('L') # 'L' คือแปลงเป็นขาวดำ (Grayscale)
        
    # 2. ปรับขนาดภาพตามที่โมเดลต้องการ
    img = img.resize(target_size)
    
    # 3. แปลงภาพเป็นอาร์เรย์ตัวเลข
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # 4. ทำ Normalization (หาร 255.0 เพื่อปรับค่าให้อยู่ในช่วง 0-1)
    img_array = img_array / 255.0
    
    # 5. เพิ่มมิติเดโม (Batch Size) ด้านหน้าให้กลายเป็น (1, Height, Width, Channels)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def is_relevant_image(image):
    """ ฟังก์ชันตรวจสอบว่ารูปที่อัปโหลดเกี่ยวข้องกับสไลด์เลือดไหม """
    img_np = np.array(image.convert('RGB'))
    mean_color = img_np.mean()
    
    # ดักจับรูปที่ขาวล้วน ดำล้วน หรือสว่าง/มืด เกินไป (ไม่ใช่รูปเม็ดเลือดแน่ๆ)
    if mean_color < 15 or mean_color > 240: 
        return False
    return True

def predict_malaria(img_array):
    """ ทำนายผลจากโมเดล Keras แบบยืดหยุ่นเพื่อป้องกัน ValueError """
    prediction = model.predict(img_array)
    
    # ตรวจสอบโครงสร้าง Output Shape ของโมเดลมึงเพื่อเลือกประมวลผลให้ถูกวิธี
    if prediction.shape[-1] == 1:
        # เคสที่ 1: โมเดลเป็นแบบ Binary (Sigmoid) พ่นค่าตัวเลขเดี่ยวๆ ออกมาช่วง 0-1
        prob = prediction[0][0]
        if prob >= 0.5:
            return "ติดเชื้อมาลาเรีย (Parasitized)", prob
        else:
            return "ไม่ติดเชื้อ (Uninfected)", 1 - prob
            
    else:
        # เคสที่ 2: โมเดลเป็นแบบ Categorical (Softmax) พ่นค่าออกมาเป็นกลุ่มคลาส (2 คลาส)
        # สมมติว่าใน Dataset มึง คลาส 0 = ปรกติ (Uninfected), คลาส 1 = ติดเชื้อ (Parasitized)
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]
        
        if class_idx == 1:
            return "ติดเชื้อมาลาเรีย (Parasitized)", confidence
        else:
            return "ไม่ติดเชื้อ (Uninfected)", confidence

# ==========================================
# 4. USER INTERFACE (ส่วนการแสดงผล)
# ==========================================
st.title("🔬 ระบบตรวจจับและคัดกรองเชื้อมาลาเรียจากภาพถ่ายเม็ดเลือด")
st.write("แอปพลิเคชันเว็บอินเตอร์เฟสสำหรับทดสอบโมเดล Deep Learning ในงานวิจัย")
st.markdown("---")

if model_loaded:
    source_image = None

    # ส่วนที่ 4.1: เลือกรูปภาพจากโฟลเดอร์ Samples
    st.subheader("📁 1. เลือกจากรูปภาพตัวอย่างในงานวิจัย (GitHub Samples)")
    if os.path.exists(SAMPLE_DIR):
        sample_files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        if sample_files:
            selected_sample = st.selectbox("เลือกไฟล์ภาพตัวอย่าง:", ["-- เลือกรูปภาพ --"] + sample_files)
            if selected_sample != "-- เลือกรูปภาพ --":
                sample_img_path = os.path.join(SAMPLE_DIR, selected_sample)
                source_image = Image.open(sample_img_path)
        else:
            st.warning(f"⚠️ ไม่พบไฟล์รูปภาพในโฟลเดอร์ '{SAMPLE_DIR}'")
    else:
        st.info(f"💡 ระบบไม่พบโฟลเดอร์ '{SAMPLE_DIR}' บนโฮสต์นี้ (มึงสามารถใช้วิธีอัปโหลดภาพแทนได้เลย)")

    # ส่วนที่ 4.2: อัปโหลดรูปภาพใหม่เอง
    st.subheader("📤 2. หรืออัปโหลดรูปภาพใหม่ที่มึงต้องการทดสอบ")
    uploaded_file = st.file_uploader("อัปโหลดไฟล์ภาพเซลล์เม็ดเลือดแดง...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        source_image = Image.open(uploaded_file)

    # ==========================================
    # 5. PROCESSING & OUTPUT (ประมวลผลและโชว์ผล)
    # ==========================================
    if source_image is not None:
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(source_image, caption="📷 รูปภาพอินพุต", use_container_width=True)
            
        with col2:
            st.subheader("📊 ผลการวิเคราะห์จากระบบ")
            
            # ขั้นดักจับ: รูปภาพไม่เกี่ยวข้อง
            if not is_relevant_image(source_image):
                st.error("❌ **รูปภาพนี้ไม่เกี่ยวข้องกับงานวิจัย**")
                st.write("คำอธิบาย: ระบบตรวจพบว่าภาพนี้ไม่ใช่ภาพถ่ายเซลล์เม็ดเลือดแดง กรุณาใช้รูปภาพที่ถูกต้องในการทดสอบ")
            
            # ขั้นประมวลผลด้วยโมเดล
            else:
                try:
                    with st.spinner("โมเดลกำลังวิเคราะห์ภาพ..."):
                        processed_img = preprocess_image(source_image)
                        result_label, confidence = predict_malaria(processed_img)
                    
                    # แสดงหน้าต่างผลลัพธ์ตามคลาสที่จำแนกได้
                    if "ติดเชื้อ" in result_label:
                        st.error(f"🚨 **ผลลัพธ์: {result_label}**")
                        st.metric(label="ค่าความเชื่อมั่น (Confidence)", value=f"{confidence * 100:.2f}%")
                        st.write("⚠️ พบลักษณะการติดเชื้อพลาสมอเดียม (Plasmodium) ในเซลล์เม็ดเลือดแดง")
                    else:
                        st.success(f"✅ **ผลลัพธ์: {result_label}**")
                        st.metric(label="ค่าความเชื่อมั่น (Confidence)", value=f"{confidence * 100:.2f}%")
                        st.write("👍 ไม่พบสิ่งผิดปกติ เซลล์เม็ดเลือดแดงอยู่ในเกณฑ์ปกติ (Uninfected)")
                        
                except Exception as eval_error:
                    st.error("❌ **เกิดข้อผิดพลาดในการรันโมเดล (Prediction Error)**")
                    st.code(f"{eval_error}")
                    st.info("💡 **วิธีแก้:** ไปปรับค่า `TARGET_IMAGE_SIZE` (บรรทัดที่ 20) หรือ `IS_COLOR_IMAGE` (บรรทัดที่ 23) ให้ตรงกับโมเดลตอนมึงสั่งเทรนในงานวิจัย!")
