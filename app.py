import streamlit as st
import os
from PIL import Image
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Malaria Detection UI", layout="centered")

# --- CUSTOM CSS (เวทมนตร์เสก UI) ---
st.markdown("""
<style>
/* พื้นหลังสีฟ้า/เทาพาสเทลแบบในรูป */
.stApp {
    background-color: #dbe4f0;
}

/* ซ่อนเมนู Streamlit ให้ดูเป็น Web App จริงๆ */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* หัวข้อ (Header) - สีม่วงเข้ม */
.main-header {
    text-align: center;
    color: #463c6d;
    font-family: 'Arial Black', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    margin-top: 2rem;
    margin-bottom: 2rem;
    line-height: 1.3;
}

/* Label เล็กๆ เหนือ Input */
.input-label {
    font-size: 0.9rem;
    color: #2c3e50;
    font-weight: 600;
    margin-bottom: 5px;
    text-transform: uppercase;
}

/* กล่องการ์ดสีขาว (Card) */
.white-card {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    margin-bottom: 20px;
    text-align: center;
}

/* คำอธิบายใต้ภาพ */
.img-caption {
    font-size: 0.8rem;
    color: #34495e;
    font-weight: 600;
    margin-top: 10px;
}

/* ปุ่มสีม่วงพาสเทล (Button) */
div.stButton > button {
    width: 100%;
    background-color: #a497d5 !important;
    color: white !important;
    border: none !important;
    border-radius: 30px !important;
    padding: 15px 0 !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 6px rgba(164, 151, 213, 0.4) !important;
    transition: 0.3s;
}
div.stButton > button:hover {
    background-color: #8a7abf !important;
}

/* กล่องผลลัพธ์ (Result Cards) */
.result-box {
    background: linear-gradient(180deg, #ffffff 40%, #95d4e8 100%);
    border-radius: 12px;
    padding: 20px 10px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    border: 1px solid #e1e8ed;
    height: 100%;
}
.result-box.purple-grad {
    background: linear-gradient(180deg, #ffffff 40%, #b2a8db 100%);
}
.result-title {
    font-size: 0.8rem;
    color: #34495e;
    font-weight: 600;
    margin-bottom: 10px;
    text-transform: uppercase;
}
.result-value {
    font-size: 1.4rem;
    font-weight: 800;
    color: #2c3e50;
    margin: 0;
}
.result-sub {
    font-size: 0.9rem;
    color: #2c3e50;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# --- 1. HEADER ---
st.markdown("""
<div class="main-header">
    LIGHTWEIGHT IMAGE CLASSIFICATION FOR<br>MALARIA DETECTION USING MOBILENETV2
</div>
""", unsafe_allow_html=True)

# --- 2. CONTROLS (SELECTION & DATASET) ---
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="input-label">SELECTION MODE</div>', unsafe_allow_html=True)
    # ใช้ Radio แนบไปเลย streamlit มันจัดการให้เป็นแนวนอนได้
    mode = st.radio("Mode", ["Samples", "Upload"], horizontal=True, label_visibility="collapsed")

with col2:
    st.markdown('<div class="input-label">CHOOSE DATASET</div>', unsafe_allow_html=True)
    img_to_predict = None
    
    if mode == "Samples":
        # จำลองการเลือกไฟล์ Sample
        sample_choice = st.selectbox("Samples", ["Image_001.jpg", "Image_002.jpg"], label_visibility="collapsed")
        # ตรงนี้มึงไปผูก logic โหลดรูปจากโฟลเดอร์ของมึงเองนะ
        # img_to_predict = Image.open(f"samples/{sample_choice}")
        st.info("💡 (ใส่รูปตัวอย่างเพื่อทดสอบระบบ)")
    else:
        uploaded_file = st.file_uploader("Upload", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
        if uploaded_file is not None:
            img_to_predict = Image.open(uploaded_file)

# --- 3. IMAGE PREVIEW ---
# กูทำกล่องขาวๆ ไว้รอรับรูป ถ้ามีรูปลงมาก็โชว์
st.markdown('<div class="white-card">', unsafe_allow_html=True)
if img_to_predict is not None:
    st.image(img_to_predict, use_container_width=True)
    st.markdown('<div class="img-caption">[ SELECTED RED BLOOD CELL MICROSCOPE IMAGE ]</div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="height: 200px; display: flex; align-items: center; justify-content: center; color: #a0aec0;">[ กรุณาเลือกหรืออัปโหลดรูปภาพเซลล์เม็ดเลือดแดง ]</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- 4. ACTION BUTTON ---
# ทำปุ่มกดยาวๆ ตามรูปเป๊ะ
analyze_clicked = st.button("START MALARIA DIAGNOSIS ANALYSIS")

# --- 5. RESULTS ---
st.markdown('<div class="input-label" style="margin-top: 20px;">Analysis result</div>', unsafe_allow_html=True)

# ถ้ากดปุ่มแล้ว ค่อยโชว์ผล
if analyze_clicked:
    # --- ตรงนี้คือที่ที่มึงต้องเอาโมเดล MobileNetV2 มารัน Predict ---
    # สมมติผลลัพธ์ว่าติดเชื้อละกัน
    mock_status = "POSITIVE"
    mock_sub = "(Malaria Infected)"
    mock_confidence = "98.45%"
    
    # แบ่ง 2 คอลัมน์เพราะตัด Analysis Quality ออกไปแล้ว
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.markdown(f"""
        <div class="result-box">
            <div class="result-title">⊕ DIAGNOSIS RESULT</div>
            <p class="result-value">{mock_status}</p>
            <p class="result-sub">{mock_sub}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with res_col2:
        st.markdown(f"""
        <div class="result-box purple-grad">
            <div class="result-title">% CONFIDENCE LEVEL</div>
            <p class="result-value" style="margin-top: 10px;">{mock_confidence}</p>
        </div>
        """, unsafe_allow_html=True)

    # --- 6. GRAD-CAM ---
    st.markdown('<div class="input-label" style="margin-top: 30px;">GRAD-CAM VISUALIZATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    
    # โชว์รูป Grad-CAM ตรงนี้ (ถ้ามึงเขียนฟังก์ชันเสร็จแล้วเอาตัวแปรภาพมาใส่แทน placeholder นี้)
    st.markdown('<div style="height: 150px; background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white;">[ รูปภาพ Heatmap ของโมเดลจะโชว์ตรงนี้ ]</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="img-caption">[ HEATMAP INDICATES AREAS OF POSSIBLE INFECTION ]</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
