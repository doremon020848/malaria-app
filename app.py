import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

MODEL_PATH = "best_model (1).keras"
IMG_SIZE   = (224, 224)

st.set_page_config(
    page_title="Malaria Cell Classification",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg:      #f7f8fa;
    --white:   #ffffff;
    --border:  #e2e6ec;
    --txt:     #1a1f2e;
    --muted:   #8691a6;
    --green:   #0f9973;
    --green-bg:#edf8f5;
    --green-b: #b2dfd4;
    --red:     #d63b52;
    --red-bg:  #fdf0f2;
    --red-b:   #f0bfc6;
    --blue:    #2563eb;
    --r:       10px;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 2rem 5rem !important; max-width: 680px !important; }

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans Thai', sans-serif !important;
    background: var(--bg) !important;
    color: var(--txt) !important;
}

/* ── HEADER ── */
.app-header {
    padding: 2rem 0 1.8rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.app-header h1 {
    font-size: 1.35rem;
    font-weight: 500;
    letter-spacing: -.01em;
    margin: 0 0 .25rem;
    color: var(--txt);
}
.app-header p {
    font-size: .82rem;
    color: var(--muted);
    margin: 0;
    font-weight: 300;
}

/* ── STATUS ── */
.status-row {
    display: flex;
    align-items: center;
    gap: .5rem;
    margin-bottom: 1.6rem;
    font-size: .78rem;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
}
.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--green);
}

/* ── CARD ── */
.card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1.4rem 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: .72rem;
    font-weight: 500;
    color: var(--muted);
    letter-spacing: .1em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── META TABLE ── */
.meta-table { width: 100%; border-collapse: collapse; }
.meta-table td {
    padding: .45rem 0;
    font-size: .84rem;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
}
.meta-table tr:last-child td { border-bottom: none; }
.meta-table td:first-child { color: var(--muted); width: 42%; font-size: .78rem; }
.meta-table td:last-child  { font-family: 'IBM Plex Mono', monospace; font-size: .82rem; }

/* ── UPLOAD ── */
[data-testid="stFileUploader"] {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
}

/* ── BUTTON ── */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: var(--txt) !important;
    color: #fff !important;
    font-family: 'IBM Plex Sans Thai', sans-serif !important;
    font-weight: 500 !important;
    font-size: .9rem !important;
    border: none !important;
    border-radius: var(--r) !important;
    padding: .75rem 1.5rem !important;
    letter-spacing: .01em !important;
    transition: opacity .18s !important;
}
[data-testid="stButton"] > button:hover { opacity: .82 !important; }

/* ── IMAGE ── */
[data-testid="stImage"] img {
    border-radius: var(--r) !important;
    border: 1px solid var(--border) !important;
}

/* ── RESULT ── */
.result-safe {
    background: var(--green-bg);
    border: 1px solid var(--green-b);
    border-radius: var(--r);
    padding: 1.6rem;
    margin-bottom: 1rem;
}
.result-danger {
    background: var(--red-bg);
    border: 1px solid var(--red-b);
    border-radius: var(--r);
    padding: 1.6rem;
    margin-bottom: 1rem;
}
.result-label {
    font-size: .72rem;
    font-weight: 500;
    letter-spacing: .1em;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: .35rem;
}
.result-title {
    font-size: 1.6rem;
    font-weight: 500;
    letter-spacing: -.01em;
    margin: 0 0 1rem;
}
.bar-track {
    height: 5px;
    background: rgba(0,0,0,.08);
    border-radius: 999px;
    overflow: hidden;
    margin: .4rem 0 .5rem;
}
.bar-safe   { height:100%; border-radius:999px; background: var(--green); }
.bar-danger { height:100%; border-radius:999px; background: var(--red);   }
.conf-row {
    display: flex;
    justify-content: space-between;
    font-size: .78rem;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
}

/* ── SPINNER ── */
[data-testid="stSpinner"] > div { color: var(--txt) !important; }

/* ── FOOTER ── */
.app-footer {
    border-top: 1px solid var(--border);
    margin-top: 2.5rem;
    padding-top: 1.2rem;
    font-size: .75rem;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🔬 Malaria Cell Classification</h1>
    <p>อัปโหลดภาพเซลล์เม็ดเลือดแดงเพื่อตรวจสอบการติดเชื้อมาลาเรีย</p>
</div>
""", unsafe_allow_html=True)

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_my_model()
    st.markdown("""
    <div class="status-row">
        <span class="status-dot"></span> Model loaded &nbsp;·&nbsp; MobileNetV2
    </div>
    """, unsafe_allow_html=True)
except Exception:
    st.error(f"ไม่พบไฟล์ `{MODEL_PATH}` — วางไว้ในโฟลเดอร์เดียวกันด้วยนะ")
    st.stop()

# ── UPLOAD ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "เลือกไฟล์ภาพ (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    img  = Image.open(uploaded_file).convert("RGB")
    w, h = img.size

    col_img, col_meta = st.columns([3, 2], gap="medium")

    with col_img:
        st.image(img, use_column_width=True)

    with col_meta:
        st.markdown(f"""
        <div class="card" style="margin-top:.1rem">
            <div class="card-title">ข้อมูลไฟล์</div>
            <table class="meta-table">
                <tr><td>ชื่อไฟล์</td>   <td>{uploaded_file.name}</td></tr>
                <tr><td>ขนาด</td>       <td>{uploaded_file.size/1024:.1f} KB</td></tr>
                <tr><td>ความละเอียด</td><td>{w} × {h} px</td></tr>
                <tr><td>Model input</td> <td>224 × 224 × 3</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # ── PREPROCESS ────────────────────────────────────────────────────────────
    img_resized = img.resize(IMG_SIZE)
    img_array   = image.img_to_array(img_resized)
    img_array   = np.expand_dims(img_array, axis=0)
    img_array   = preprocess_input(img_array)

    if st.button("วิเคราะห์ผล"):
        with st.spinner("กำลังประมวลผล..."):
            prediction = float(model.predict(img_array)[0][0])

        confidence = prediction if prediction > 0.5 else 1 - prediction
        conf_pct   = confidence * 100
        is_safe    = prediction > 0.5

        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

        if is_safe:
            st.balloons()
            st.markdown(f"""
            <div class="result-safe">
                <div class="result-label" style="color:var(--green)">ผลการวิเคราะห์</div>
                <div class="result-title" style="color:var(--green)">ไม่พบการติดเชื้อ</div>
                <div class="bar-track"><div class="bar-safe" style="width:{conf_pct:.1f}%"></div></div>
                <div class="conf-row">
                    <span>Confidence</span>
                    <span>{conf_pct:.2f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-danger">
                <div class="result-label" style="color:var(--red)">ผลการวิเคราะห์</div>
                <div class="result-title" style="color:var(--red)">ตรวจพบการติดเชื้อ</div>
                <div class="bar-track"><div class="bar-danger" style="width:{conf_pct:.1f}%"></div></div>
                <div class="conf-row">
                    <span>Confidence</span>
                    <span>{conf_pct:.2f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Raw output"):
            st.code(f"""sigmoid  = {prediction:.8f}
threshold = 0.50000000
class     = {"Uninfected (1)" if is_safe else "Infected (0)"}
conf      = {conf_pct:.4f}%""", language="text")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <span>Data Science · CMU</span>
    <span>MobileNetV2 · TensorFlow</span>
</div>
""", unsafe_allow_html=True)
