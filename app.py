import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

# =================================================================
# ─── CONFIGURATION (ตั้งค่าตรงนี้เลยสัส) ───────────────────────────
# =================================================================
# 1. ใส่ชื่อไฟล์โมเดลของมึง (เช็คชื่อไฟล์ดีๆ ว่าตรงกับในเครื่องไหม)
MODEL_PATH = "malaria_mobilenetv2_model (2).keras" 

# 2. ใส่พาธของรูปภาพเม็ดเลือดที่มึงต้องการจะสั่งให้มันทาย
IMAGE_TO_PREDICT = "samples/ชื่อไฟล์รูปภาพของมึง.png" 

IMG_SIZE = (224, 224)

# =================================================================
# ─── SYSTEM CORE (ระบบรันโหลดโมเดลและทายผล) ────────────────────────
# =================================================================

def predict_single_image(model_file, img_path):
    # 1. ตรวจสอบไฟล์ก่อนรัน ป้องกันโค้ดเอ๋อแดก
    if not os.path.exists(model_file):
        print(f"❌ ERROR: ไม่พบไฟล์โมเดล '{model_file}' ในโฟลเดอร์นี้สลัด!")
        return
    if not os.path.exists(img_path):
        print(f"❌ ERROR: ไม่พบไฟล์รูปภาพ '{img_path}' เช็คพาธและชื่อไฟล์ด่วน!")
        return

    print("⏳ กำลังโหลดสมองกล AI...")
    # โหลดโมเดลแบบปิด compile ไว้เพื่อความเร็วและลด error เรื่องเวอร์ชั่น Keras
    model = tf.keras.models.load_model(model_path, compile=False)
    print("✅ โหลดโมเดลสำเร็จ!")

    # 2. ทำ Preprocess รูปภาพแปลงเข้าสู่มิติที่โมเดลต้องการ [1, 224, 224, 3]
    print(f"📷 กำลังประมวลผลรูปภาพ: {os.path.basename(img_path)}")
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array) # สเกลค่าพิกเซลให้เข้ากับ MobileNetV2

    # 3. สั่งโมเดลลุยทำนายผลลัพธ์ (Predict)
    print("🚀 AI กำลังส่องกล้องวิเคราะห์หาเชื้อมาลาเรีย...")
    predictions = model.predict(img_array)[0] # ดึงผลลัพธ์รูปแรกออกมา

    # 4. ใช้ np.argmax หาตัวเลขคลาสที่ทำคะแนนได้สูงสุดจริงๆ (แก้บั๊กทายซ้ำซาก)
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]

    # 5. แยกเงื่อนไขสรุปคำตอบตามที่มึงบรีฟไว้
    if predicted_class == 0:
        status = "🚨 เซลล์ติดเชื้อ (INFECTED DETECTED)"
    else:
        status = "✅ เซลล์ปกติ (NORMAL CELL)"

    # 6. พ่นผลลัพธ์ความหล่อออกมาทางหน้าจอ Console
    print("\n" + "="*50)
    print("📊 [ ผลการตรวจวิเคราะห์จากระบบ ]")
    print(f"➡️ ผลลัพธ์: {status}")
    print(f"➡️ ค่าความมั่นใจ (Confidence Level): {confidence * 100:.2f}%")
    print("="*50 + "\n")


# ─── สั่งรันฟังก์ชัน ──────────────────────────────────────────────────
if __name__ == "__main__":
    predict_single_image(MODEL_PATH, IMAGE_TO_PREDICT)
