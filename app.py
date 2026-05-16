import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# ==========================================
# 1. CONFIG & SETTINGS (ตั้งค่าหน้าเว็บ)
# ==========================================
st.set_page_config(
    page_title="Malaria Detection Research App",
    page_icon="🔬",
    layout="centered"
)

# 📌 [จุดสำคัญ] ปรับแก้ชื่อไฟล์โมเดล และ ขนาดภาพ (Target Size) ให้ตรงกับงานวิจัยของมึง
MODEL_PATH = "best_model (6).keras"
TARGET_IMAGE_SIZE = (224, 224)  # ถ้าตอนเทรนใช้ 100x100 หรือ 128x128 ให้มาแก้ตรงนี้เลยมึง!
SAMPLE_DIR = "samples"          # ชื่อโฟลเดอร์รูปตัวอย่างที่ดึงมาจาก GitHub

# ==========================================
# 2. LOAD MODEL (โหลดโมเดลเข้าแคช)
# ==========================================
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_my_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ โหลดโมเดลไม่สำเร็จ! เช็คว่าไฟล์ '{MODEL_PATH}' อยู่ในไดเรกทอรีเดียวกับโค้ดนี้ไหม")
    st.info(f"รายละเอียด Error: {e}")

# ==========================================
# 3. HELPER FUNCTIONS (ฟังก์ชันช่วยประมวลผล)
# ==========================================
def preprocess_image(image, target_size=TARGET_IMAGE_SIZE):
    """ แปลงรูปภาพให้อยู่ในมิติ (Shape) ที่โมเดลต้องการ และทำ Normalization """
    # 1. ปรับขนาดภาพให้เท่ากับที่โมเดลถูกเทรนมา
    img = image.resize(target_size)
    
    # 2. แปลงภาพเป็นอาร์เรย์ (Height, Width, Channels)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # 3. ทำ Normalization (หาร 255.0 เพื่อปรับค่าพิกเซลให้อยู่ในช่วง 0-1)
    img_array = img_array / 255.0
    
    # 4. เพิ่มมิติเดโม (Batch Size) ให้กลายเป็น (1, Height, Width, Channels) ตามมาตรฐาน Keras
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def is_relevant_image(image):
    """ 
    ฟังก์ชันตรวจสอบความเกี่ยวข้องเบื้องต้น 
    ดักพวกรูปหมา รูปแมว หรือรูปทั่วไปที่ไม่ใช่สไลด์ส่องเม็ดเลือดแดง
    """
    img_np = np.array(image.convert('RGB'))
    mean_color = img_np.mean()
    
    # ดักจับรูปที่ขาวล้วน ดำล้วน หรือสว่าง/มืด เกินไป (มึงไปปรับจูนค่าพวกนี้ตาม Dataset มึงได้)
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
        # เคสที่ 2: โมเดลเป็นแบบ Categorical (Softmax) พ่นค่าออกมาเป็นกลุ่มคลาส (เช่น 2 คลาส)
        # สมมติว่าใน Dataset มึง คลาส 0 = ปรกติ (Uninfected), คลาส 1 = ติดเชื้อ (Parasitized)
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]
        
        if class_idx == 1:
            return "ติดเชื้อมาลาเรีย (Parasitized)", confidence
        else:
            return "ไม่ติดเชื้อ (Uninfected)", confidence

# ==========================================
# 4. USER INTERFACE (ส่วนแสดงผลหน้าเว็บ)
# ==========================================
st.title("🔬 ระบบตรวจจับและคัดกรองเชื้อมาลาเรียจากภาพถ่ายเม็ดเลือด")
st.write("แอปพลิเคชันเว็บอินเตอร์เฟสสำหรับจำลองผลงานวิจัยด้วยโมเดล Deep Learning")
st.markdown("---")

if model_loaded:
    source_image = None

    # ส่วนที่ 4.1: เลือกรูปภาพจากโฟลเดอร์ Samples ใน GitHub
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
        st.info(f"💡 ระบบไม่พบโฟลเดอร์ '{SAMPLE_DIR}' บนเครื่องโลคอลนี้ (มึงสามารถใช้วิธีอัปโหลดภาพแทนได้)")

    # ส่วนที่ 4.2: อัปโหลดรูปภาพใหม่เอง
    st.subheader("📤 2. หรืออัปโหลดรูปภาพใหม่ที่มึงต้องการทดสอบ")
    uploaded_file = st.file_uploader("อัปโหลดไฟล์ภาพเซลล์เม็ดเลือดแดง...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        source_image = Image.open(uploaded_file)

    # ==========================================
    # 5. PROCESSING & OUTPUT (ประมวลผลและโชว์ผลลัพธ์)
    # ==========================================
    if source_image is not None:
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(source_image, caption="📷 รูปภาพอินพุตที่จะทำการวิเคราะห์", use_container_width=True)
            
        with col2:
            st.subheader("📊 ผลการวิเคราะห์จากระบบ")
            
            # ขั้นตอนดักจับ: เช็คว่ารูปเกี่ยวข้องกับงานวิจัยหรือไม่
            if not is_relevant_image(source_image):
                st.error("❌ **รูปภาพนี้ไม่เกี่ยวข้องกับงานวิจัย**")
                st.write("ข้อความจากระบบ: ตรวจพบว่าภาพไม่ใช่สไลด์เซลล์เม็ดเลือดแดง กรุณาใช้รูปภาพที่ถูกต้องในการทดสอบ")
            
            else:
                # ขั้นตอนทำนาย: ส่งเข้าโมเดล Keras
                try:
                    with st.spinner("โมเดลกำลังประมวลผลและจำแนกภาพ..."):
                        processed_img = preprocess_image(source_image, target_size=TARGET_IMAGE_SIZE)
                        result_label, confidence = predict_malaria(processed_img)
                    
                    # แสดงหน้าต่างผลลัพธ์ตามคลาสที่ได้
                    if "ติดเชื้อ" in result_label:
                        st.error(f"🚨 **ผลลัพธ์: {result_label}**")
                        st.metric(label="ค่าความเชื่อมั่น (Confidence)", value=f"{confidence * 100:.2f}%")
                        st.write("⚠️ ตรวจพบสิ่งเจือปนหรือเชื้อพลาสมอเดียมภายในเซลล์เม็ดเลือดแดง")
                    else:
                        st.success(f"✅ **ผลลัพธ์: {result_label}**")
                        st.metric(label="ค่าความเชื่อมั่น (Confidence)", value=f"{confidence * 100:.2f}%")
                        st.write("👍 เซลล์เม็ดเลือดแดงมีความสมบูรณ์และไม่มีการติดเชื้อมาลาเรีย")
                        
                except Exception as eval_error:
                    st.error("❌ **เกิดข้อผิดพลาดระหว่างการคำนวณของโมเดล (Prediction Error)**")
                    st.code(f"Error Code: {eval_error}")
                    st.info("💡 คำแนะนำ: ลองเช็คค่า `TARGET_IMAGE_SIZE` ในโค้ดบรรทัดที่ 16 ว่าตรงกับตอนที่มึงสั่ง Train โมเดลหรือไม่!")
