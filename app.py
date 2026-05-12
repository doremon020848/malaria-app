import streamlit as st

# 1. Header
st.title("MobileNetV2-based Classification of Malaria-Infected Red Blood Cells")

# 2. Sidebar
with st.sidebar:
    st.header("Settings")
    model_name = "MobileNetV2"
    mode = st.radio("Select Mode", ["Sample Images", "Upload Image"])
    st.info(f"Current Model: {model_name}")

# 3. Main Content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Image")
    # โชว์รูปที่เลือก
    st.image(input_img, caption="Selected Cell Image")

with col2:
    st.subheader("Analysis Result")
    if st.button("Run Prediction"):
        # ส่วนเรียกโมเดล Predict
        result = "Infected" # สมมติ
        confidence = 0.98
        
        # แสดงผลตามเงื่อนไข
        if result == "Infected":
            st.error(f"Status: {result}")
        else:
            st.success(f"Status: {result}")
            
        st.write(f"Confidence Score: {confidence*100:.2f}%")
        st.progress(confidence)
