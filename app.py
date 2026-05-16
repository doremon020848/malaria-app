import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- 1. CONFIG & SETTINGS ---
st.set_page_config(
    page_title="Malaria Detection & Classification",
    page_icon="🔬",
    layout="centered"
)

MODEL_PATH = "best_model (6).keras"
# โฟลเดอร์รูปตัวอย่างที่ดึงมาจาก GitHub (ถ้าดึงมาโลคอลหรือดึงผ่าน URL ก็ปรับให้ตรง)
SAMPLE_DIR = "samples" 

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_my_model():
    # โหลดโมเดล Keras ของมึง
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_my_model()
except Exception as e:
    st.error(f"❌ โหลดโมเดลไม่สำเร็จ เช็คชื่อไฟล์หรือ Path ดีๆ: {e}")
    st.stop()

# --- 3. HELPER FUNCTIONS ---
def preprocess_image(image, target_size=(224, 224)):
    """ แปลงรูปภาพให้อยู่ใน Format ที่โมเดลพร้อมใช้งาน """
    # ปรับขนาดรูปให้เท่ากับที่โมเดลต้องการ (เช็คตามสถาปัตยกรรมที่มึงเทรนมา เช่น 224x224)
    img = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # ทำ Normalization (เช็คด้วยว่าตอนเทรนใช้ /255.0 หรือเปล่า)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0) # (1, 224, 224, 3)
    return img_array

def is_relevant_image(image):
    """ 
    ฟังก์ชันตรวจสอบว่ารูปที่อัปโหลดเกี่ยวข้องกับเซลล์เม็ดเลือดไหม 
    (มึงสามารถใช้เกณฑ์ เช่น เช็คค่าเฉลี่ยสี, ความสว่าง หรือสร้างโมเดลตัวเล็กมาดัก)
    """
    img_np = np.array(image.convert('RGB'))
    
    # ตัวอย่างการดักแบบง่าย: เช็คค่าเฉลี่ยสี (เม็ดเลือดแดงส่วนใหญ่จะมีโทนสีจำเพาะ)
    # หรือเช็ค Resolution/ความหลากหลายของสี ถ้าเป็นรูปวิว รูปหมาแมว ค่าสถิติจะต่างออกไป
    # *แนะนำให้ปรับเงื่อนไขนี้ตามคุณลักษณะของ Dataset มึง*
    mean_color = img_np.mean()
    
    # สมมติว่าถ้ารูปมืดไป สว่างไป หรือไม่ใช่โทนสีของสไลด์เลือด ให้ตีว่าไม่เกี่ยวข้อง
    if mean_color < 10 or mean_color > 245: 
        return False
        
    return True

def predict_malaria(img_array):
    """ ทำนายผลจากโมเดล """
    prediction = model.predict(img_array)
    
    # สมมติว่าโมเดลเอาต์พุตเป็น Binary (Sigmoid) โอกาสติดเชื้ออยู่ตำแหน่งที่ 0
    # หรือถ้าเป็น Softmax 2 Classes ให้ปรับตามโครงสร้างเอาต์พุตของมึงนะ
    prob = prediction[0][0] 
    
    # สมมติ: 0 = ปรกติ (Uninfected), 1 = ติดเชื้อ (Parasitized)
    if prob >= 0.5:
        return "ติดเชื้อมาลาเรีย (Parasitized)", prob
    else:
        return "ไม่ติดเชื้อ (Uninfected)", 1 - prob

# --- 4. USER INTERFACE (UI) ---
st.title("🔬 ระบบคัดกรองผู้ป่วยติดเชื้อมาลาเรียจากภาพถ่ายเม็ดเลือด")
st.write("แอปพลิเคชันสำหรับงานวิจัยเปรียบเทียบประสิทธิภาพโมเดล Deep Learning")
st.markdown("---")

# ส่วนเลือกรูปตัวอย่างจาก GitHub Samples
st.subheader("📁 เลือกดูจากรูปภาพตัวอย่าง (GitHub Samples)")
if os.path.exists(SAMPLE_DIR):
    sample_files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    if sample_files:
        selected_sample = st.selectbox("เลือกรูปภาพตัวอย่าง:", ["-- เลือกรูปภาพ --"] + sample_files)
        if selected_sample != "-- เลือกรูปภาพ --":
            sample_img_path = os.path.join(SAMPLE_DIR, selected_sample)
            source_image = Image.open(sample_img_path)
    else:
        st.warning("⚠️ ไม่พบไฟล์รูปภาพในโฟลเดอร์ samples")
        source_image = None
else:
    st.info("💡 โฟลเดอร์ samples ไม่ได้อยู่บนโลคอลนี้ (ระบบจะอิงตามการอัปโหลดเป็นหลัก)")
    source_image = None

# ส่วนอัปโหลดรูปภาพใหม่
st.subheader("📤 หรืออัปโหลดรูปภาพใหม่")
uploaded_file = st.file_uploader("เลือกรูปภาพเซลล์เม็ดเลือดแดง (.png, .jpg)...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    source_image = Image.open(uploaded_file)

# --- 5. INFERENCE & DISPLAY RESULT ---
if source_image is not None:
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(source_image, caption="รูปภาพที่เลือก/อัปโหลด", use_column_width=True)
        
    with col2:
        st.subheader("📊 ผลการวิเคราะห์")
        
        # ขั้นที่ 1: ตรวจสอบความเกี่ยวข้องของรูปภาพ
        if not is_relevant_image(source_image):
            st.error("❌ **รูปภาพนี้ไม่เกี่ยวข้อง**")
            st.write("ระบบตรวจพบว่าภาพนี้ไม่ใช่ภาพถ่ายเซลล์เม็ดเลือดแดงที่ใช้ในงานวิจัย กรุณาใช้รูปภาพที่ถูกต้อง")
        
        else:
            # ขั้นที่ 2: ประมวลผลและทำนายด้วยโมเดล
            with st.spinner("กำลังประมวลผลภาพด้วยโมเดล..."):
                # ปรับ target_size ให้ตรงกับ Input Size ของโมเดล (MobileNetV2, ResNet, EfficientNet ที่มึงใช้)
                processed_img = preprocess_image(source_image, target_size=(224, 224))
                result_label, confidence = predict_malaria(processed_img)
                
            # แสดงผลลัพธ์ตามคลาส
            if "ติดเชื้อ" in result_label:
                st.metric(label="ผลลัพธ์", value="🚨 ติดเชื้อ (Parasitized)")
                st.warning(f"ความเชื่อมั่น (Confidence): {confidence * 100:.2f}%")
            else:
                st.metric(label="ผลลัพธ์", value="✅ ปกติ (Uninfected)")
                st.success(f"ความเชื่อมั่น (Confidence): {confidence * 100:.2f}%")
