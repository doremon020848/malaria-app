import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
MODEL_PATH = "best_model (1).keras"
IMG_SIZE = (224, 224)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Malaria Detection · CMU",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ─── RESET & ROOT ─────────────────────────────────────────────── */
:root {
    --bg:        #0b0f14;
    --bg2:       #111720;
    --bg3:       #182030;
    --border:    #1e2d42;
    --accent:    #00d4aa;
    --accent2:   #0096ff;
    --danger:    #ff4f6d;
    --warn:      #ffb830;
    --text:      #e2ecf7;
    --muted:     #6b8aaa;
    --card-r:    16px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 2rem 4rem !important;
    max-width: 780px !important;
}

/* ─── HERO HEADER ──────────────────────────────────────────────── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
}
.hero-icon {
    font-size: 3.5rem;
    display: block;
    margin-bottom: .5rem;
    filter: drop-shadow(0 0 24px #00d4aa88);
    animation: pulse 3s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { filter: drop-shadow(0 0 16px #00d4aa66); }
    50%       { filter: drop-shadow(0 0 36px #00d4aacc); }
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    font-weight: 400;
    letter-spacing: -0.02em;
    margin: 0 0 .4rem;
    background: linear-gradient(135deg, #e2ecf7 30%, #00d4aa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    color: var(--muted);
    font-size: .95rem;
    font-weight: 300;
    margin: 0;
    letter-spacing: .02em;
}

/* ─── STATUS BADGE ─────────────────────────────────────────────── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    background: #00d4aa18;
    border: 1px solid #00d4aa44;
    border-radius: 999px;
    padding: .3rem .9rem;
    font-size: .78rem;
    font-family: 'DM Mono', monospace;
    color: var(--accent);
    margin: 1rem auto .5rem;
    letter-spacing: .04em;
}
.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
    animation: blink 1.6s ease infinite;
}
@keyframes blink {
    0%,100% { opacity: 1; }
    50%      { opacity: .3; }
}

/* ─── CARDS ────────────────────────────────────────────────────── */
.card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--card-r);
    padding: 1.6rem;
    margin-bottom: 1.2rem;
    transition: border-color .25s;
}
.card:hover { border-color: #2a4060; }

.card-label {
    font-size: .72rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: .12em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: .8rem;
}

/* ─── RESULT PANEL ─────────────────────────────────────────────── */
.result-safe {
    background: linear-gradient(135deg, #00d4aa08, #0096ff08);
    border: 1px solid #00d4aa33;
    border-radius: var(--card-r);
    padding: 1.8rem;
    text-align: center;
}
.result-danger {
    background: linear-gradient(135deg, #ff4f6d08, #ffb83008);
    border: 1px solid #ff4f6d44;
    border-radius: var(--card-r);
    padding: 1.8rem;
    text-align: center;
}
.result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    margin: .4rem 0 .2rem;
}
.result-sub {
    font-size: .85rem;
    color: var(--muted);
    margin-bottom: 1rem;
}
.conf-bar-wrap {
    background: var(--bg3);
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
    margin: .6rem 0;
}
.conf-bar-fill-safe {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #00d4aa, #0096ff);
    transition: width .8s cubic-bezier(.4,0,.2,1);
}
.conf-bar-fill-danger {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #ff4f6d, #ffb830);
    transition: width .8s cubic-bezier(.4,0,.2,1);
}
.conf-num {
    font-family: 'DM Mono', monospace;
    font-size: 1.1rem;
    font-weight: 500;
}

/* ─── METRICS ROW ──────────────────────────────────────────────── */
.metrics-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.2rem;
}
.metric-box {
    flex: 1;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
.metric-val {
    font-family: 'DM Mono', monospace;
    font-size: 1.4rem;
    font-weight: 500;
    color: var(--accent);
}
.metric-lbl {
    font-size: .72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-top: .1rem;
}

/* ─── UPLOAD ZONE ──────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--bg2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--card-r) !important;
    padding: .5rem !important;
    transition: border-color .25s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: var(--muted) !important;
}

/* ─── BUTTON ───────────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #00d4aa, #0096ff) !important;
    color: #050a0f !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: .95rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: .75rem 2rem !important;
    width: 100% !important;
    letter-spacing: .02em !important;
    transition: opacity .2s, transform .15s !important;
    box-shadow: 0 4px 20px #00d4aa33 !important;
}
[data-testid="stButton"] > button:hover {
    opacity: .88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 28px #00d4aa44 !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ─── SPINNER ──────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div { color: var(--accent) !important; }

/* ─── DIVIDER ──────────────────────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ─── IMAGE ────────────────────────────────────────────────────── */
[data-testid="stImage"] img {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
}

/* ─── ALERTS ───────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-size: .88rem !important;
}

/* ─── FOOTER ───────────────────────────────────────────────────── */
.app-footer {
    text-align: center;
    padding: 2rem 0 0;
    color: var(--muted);
    font-size: .75rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: .06em;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}

/* ─── DISCLAIMER CARD ──────────────────────────────────────────── */
.disclaimer {
    background: #ffb83010;
    border: 1px solid #ffb83033;
    border-radius: 10px;
    padding: .9rem 1.1rem;
    font-size: .8rem;
    color: #ffb830cc;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: flex-start;
    gap: .6rem;
}
</style>
""", unsafe_allow_html=True)

# ─── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-icon">🔬</span>
    <h1>Malaria Cell Detection</h1>
    <p class="hero-sub">MobileNetV2 · Transfer Learning · Binary Classification</p>
</div>
""", unsafe_allow_html=True)

# ─── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_my_model()
    st.markdown("""
    <div style="text-align:center; margin-bottom:1.5rem;">
        <span class="status-badge">
            <span class="status-dot"></span>MODEL READY
        </span>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"⚠️  โหลดโมเดลไม่สำเร็จ — วางไฟล์ `{MODEL_PATH}` ไว้ในโฟลเดอร์เดียวกันด้วยนะ")
    st.stop()

# ─── DISCLAIMER ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    &nbsp;
    <span></span>
</div>
""", unsafe_allow_html=True)

# ─── UPLOAD ────────────────────────────────────────────────────────────────────
st.markdown('<div class="card-label">📁 &nbsp;อัปโหลดภาพเซลล์เม็ดเลือดแดง</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    label="วางหรือเลือกไฟล์ JPG / PNG",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    w, h = img.size

    # ─── PREVIEW + META ────────────────────────────────────────────────────────
    col_img, col_meta = st.columns([3, 2], gap="medium")
    with col_img:
        st.image(img, caption="", use_column_width=True)

    with col_meta:
        st.markdown(f"""
        <div style="margin-top:.5rem;">
            <div class="card-label">ℹ️ &nbsp;รายละเอียดไฟล์</div>
        </div>
        <div class="metrics-row" style="flex-direction:column; gap:.7rem;">
            <div class="metric-box">
                <div class="metric-val">{uploaded_file.name[:20]}</div>
                <div class="metric-lbl">ชื่อไฟล์</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{w} × {h}</div>
                <div class="metric-lbl">ความละเอียด (px)</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{uploaded_file.size / 1024:.1f} KB</div>
                <div class="metric-lbl">ขนาดไฟล์</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

    # ─── PREPROCESS ────────────────────────────────────────────────────────────
    img_resized = img.resize(IMG_SIZE)
    img_array  = image.img_to_array(img_resized)
    img_array  = np.expand_dims(img_array, axis=0)
    img_array  = preprocess_input(img_array)

    # ─── ANALYSE BUTTON ────────────────────────────────────────────────────────
    if st.button("🧬  เริ่มวิเคราะห์เซลล์"):
        with st.spinner("กำลังประมวลผลด้วย Neural Network…"):
            prediction = model.predict(img_array)[0][0]

        confidence = float(prediction if prediction > 0.5 else 1 - prediction)
        conf_pct   = confidence * 100
        is_safe    = prediction > 0.5

        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

        if is_safe:
            st.balloons()
            st.markdown(f"""
            <div class="result-safe">
                <div style="font-size:2.8rem; margin-bottom:.3rem;">✅</div>
                <div class="result-label" style="color:#00d4aa;">Uninfected</div>
                <div class="result-sub">ไม่พบการติดเชื้อมาลาเรีย</div>
                <div class="conf-bar-wrap">
                    <div class="conf-bar-fill-safe" style="width:{conf_pct:.1f}%"></div>
                </div>
                <div class="conf-num" style="color:#00d4aa;">{conf_pct:.2f}% Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-danger">
                <div style="font-size:2.8rem; margin-bottom:.3rem;">⚠️</div>
                <div class="result-label" style="color:#ff4f6d;">Infected</div>
                <div class="result-sub">ตรวจพบเชื้อมาลาเรีย — กรุณาพบแพทย์</div>
                <div class="conf-bar-wrap">
                    <div class="conf-bar-fill-danger" style="width:{conf_pct:.1f}%"></div>
                </div>
                <div class="conf-num" style="color:#ff4f6d;">{conf_pct:.2f}% Confidence</div>
            </div>
            """, unsafe_allow_html=True)

        # ─── RAW SCORE ─────────────────────────────────────────────────────────
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        with st.expander("🔢  Raw model output"):
            st.markdown(f"""
            <div style="font-family:'DM Mono',monospace; font-size:.82rem; color:var(--muted,#6b8aaa); line-height:1.9;">
                sigmoid output &nbsp;=&nbsp; <span style="color:#e2ecf7">{prediction:.6f}</span><br>
                threshold &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=&nbsp; <span style="color:#e2ecf7">0.5</span><br>
                class &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=&nbsp; <span style="color:#00d4aa">{'Uninfected (1)' if is_safe else 'Infected (0)'}</span><br>
                input size &nbsp;&nbsp;&nbsp;=&nbsp; <span style="color:#e2ecf7">224 × 224 × 3</span>
            </div>
            """, unsafe_allow_html=True)

# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    พัฒนาโดย <strong>ไอ้ยี่สิบ</strong> &nbsp;·&nbsp; Data Science · CMU &nbsp;·&nbsp; MobileNetV2 backbone
</div>
""", unsafe_allow_html=True)
