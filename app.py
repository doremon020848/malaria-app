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
    page_title="MalariaScope · CMU",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── MASTER CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Outfit:wght@300;400;500&display=swap');

:root {
    --ink:      #04080f;
    --surface:  #080e1a;
    --glass:    rgba(8,20,40,0.72);
    --border:   rgba(0,210,180,0.13);
    --border2:  rgba(0,210,180,0.28);
    --cyan:     #00d2b4;
    --cyan2:    #00ffe5;
    --blue:     #0a84ff;
    --red:      #ff3d6b;
    --amber:    #ffc043;
    --txt:      #d8eeff;
    --muted:    #4d7a99;
    --r:        18px;
}

/* ── HIDE STREAMLIT CHROME ─────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 1.8rem 5rem !important;
    max-width: 800px !important;
}

/* ── GLOBAL ────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    background-color: var(--ink) !important;
    color: var(--txt) !important;
}

/* ── ANIMATED GRID BACKGROUND ──────────────────────────────────── */
.grid-bg {
    position: fixed;
    inset: 0;
    z-index: 0;
    background-image:
        linear-gradient(rgba(0,210,180,.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,210,180,.04) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
}
.grid-bg::after {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 70% 60% at 50% 0%, rgba(0,210,180,.09) 0%, transparent 70%),
        radial-gradient(ellipse 50% 40% at 80% 100%, rgba(10,132,255,.07) 0%, transparent 60%);
}

/* ── HERO ──────────────────────────────────────────────────────── */
.hero-wrap {
    position: relative;
    padding: 3.5rem 1rem 2.5rem;
    text-align: center;
}
.hero-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: .68rem;
    letter-spacing: .22em;
    color: var(--cyan);
    text-transform: uppercase;
    margin-bottom: .9rem;
    opacity: .85;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 6vw, 3.6rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -.03em;
    margin: 0 0 .6rem;
    background: linear-gradient(160deg, #ffffff 20%, var(--cyan) 60%, var(--blue) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-desc {
    font-size: .9rem;
    color: var(--muted);
    font-weight: 300;
    letter-spacing: .02em;
    max-width: 380px;
    margin: 0 auto;
    line-height: 1.7;
}
.pill {
    display: inline-flex;
    align-items: center;
    gap: .5rem;
    background: rgba(0,210,180,.08);
    border: 1px solid var(--border2);
    border-radius: 999px;
    padding: .32rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: .7rem;
    color: var(--cyan);
    letter-spacing: .1em;
    margin: 1.2rem auto 0;
}
.pill-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--cyan);
    box-shadow: 0 0 8px var(--cyan);
    animation: beat 1.8s ease-in-out infinite;
}
@keyframes beat {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:.3; transform:scale(.6); }
}

/* ── SECTION LABEL ─────────────────────────────────────────────── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: .65rem;
    letter-spacing: .18em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: .5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border2), transparent);
}

/* ── FILE INFO GRID ────────────────────────────────────────────── */
.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: .7rem;
    margin-top: .5rem;
}
.info-cell {
    background: rgba(0,210,180,.05);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: .75rem 1rem;
}
.info-val {
    font-family: 'Space Mono', monospace;
    font-size: .95rem;
    color: var(--txt);
    font-weight: 700;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.info-key {
    font-size: .68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: .1em;
    margin-top: .15rem;
}

/* ── UPLOAD ────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(0,210,180,.04) !important;
    border: 1.5px dashed rgba(0,210,180,.22) !important;
    border-radius: var(--r) !important;
    transition: border-color .3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,210,180,.55) !important;
}

/* ── ANALYSE BUTTON ────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, var(--cyan) 0%, var(--blue) 100%) !important;
    color: #040c18 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: .06em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: .9rem 2rem !important;
    box-shadow: 0 0 32px rgba(0,210,180,.25), 0 4px 20px rgba(0,0,0,.5) !important;
    transition: transform .15s, box-shadow .2s, opacity .2s !important;
    text-transform: uppercase !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 52px rgba(0,210,180,.42), 0 8px 30px rgba(0,0,0,.6) !important;
    opacity: .92 !important;
}
[data-testid="stButton"] > button:active { transform: translateY(0) !important; }

/* ── RESULT PANELS ─────────────────────────────────────────────── */
.result-box {
    border-radius: var(--r);
    padding: 2.4rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-box::before {
    content: '';
    position: absolute;
    top: -80px; left: 50%; transform: translateX(-50%);
    width: 300px; height: 300px;
    border-radius: 50%;
    filter: blur(70px);
    pointer-events: none;
    opacity: .3;
}
.result-safe {
    background: linear-gradient(160deg, rgba(0,210,180,.08), rgba(10,132,255,.05));
    border: 1px solid rgba(0,210,180,.32);
}
.result-safe::before   { background: var(--cyan); }
.result-danger {
    background: linear-gradient(160deg, rgba(255,61,107,.08), rgba(255,192,67,.04));
    border: 1px solid rgba(255,61,107,.34);
}
.result-danger::before { background: var(--red); }

.result-icon  { font-size: 3.4rem; margin-bottom: .5rem; }
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -.03em;
    margin: 0 0 .2rem;
}
.result-eng {
    font-size: .72rem;
    color: var(--muted);
    letter-spacing: .14em;
    font-family: 'Space Mono', monospace;
    margin-bottom: 1.4rem;
    text-transform: uppercase;
}
.conf-pct {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -.02em;
    line-height: 1;
}
.bar-track {
    height: 6px;
    background: rgba(255,255,255,.07);
    border-radius: 999px;
    overflow: hidden;
    margin: .7rem auto;
    max-width: 340px;
}
.bar-fill-safe   { height:100%; border-radius:999px; background: linear-gradient(90deg,var(--cyan),var(--blue)); }
.bar-fill-danger { height:100%; border-radius:999px; background: linear-gradient(90deg,var(--red),var(--amber)); }

/* image */
[data-testid="stImage"] img {
    border-radius: 14px !important;
    border: 1px solid var(--border2) !important;
}

/* spinner */
[data-testid="stSpinner"] > div { color: var(--cyan) !important; }

/* expander */
details { border: 1px solid var(--border) !important; border-radius: 12px !important; }

/* ── FOOTER ────────────────────────────────────────────────────── */
.footer {
    text-align: center;
    padding: 2.5rem 0 0;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    font-size: .65rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    border-top: 1px solid var(--border);
    margin-top: 2.5rem;
}
.footer span { color: var(--cyan); }
</style>

<div class="grid-bg"></div>
""", unsafe_allow_html=True)

# ─── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <p class="hero-eyebrow">🧬 &nbsp; Plasmodium Detection System</p>
    <h1 class="hero-title">Malaria<br>Scope</h1>
    <p class="hero-desc">วิเคราะห์เซลล์เม็ดเลือดแดงด้วย MobileNetV2<br>ความแม่นยำสูง · ประมวลผลเร็ว</p>
    <div style="text-align:center">
        <span class="pill">
            <span class="pill-dot"></span> MODEL LOADED · CMU DATA SCI
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_my_model()
except Exception:
    st.error(f"⚠️  ไม่พบไฟล์ `{MODEL_PATH}` — วางไว้ในโฟลเดอร์เดียวกันด้วยนะ")
    st.stop()

# ─── UPLOAD ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">📁 &nbsp;อัปโหลดภาพเซลล์</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    label="drag & drop หรือเลือกไฟล์ JPG / PNG",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible",
)

if uploaded_file is not None:
    img  = Image.open(uploaded_file).convert("RGB")
    w, h = img.size

    # ── IMAGE + META ──────────────────────────────────────────────────────────
    col_img, col_info = st.columns([5, 4], gap="medium")

    with col_img:
        st.image(img, use_column_width=True)

    with col_info:
        st.markdown(f"""
        <div style="height:.4rem"></div>
        <div class="section-label">ℹ️ &nbsp;ข้อมูลไฟล์</div>
        <div class="info-grid">
            <div class="info-cell" style="grid-column:1/-1">
                <div class="info-val">{uploaded_file.name}</div>
                <div class="info-key">Filename</div>
            </div>
            <div class="info-cell">
                <div class="info-val">{w}×{h}</div>
                <div class="info-key">Resolution</div>
            </div>
            <div class="info-cell">
                <div class="info-val">{uploaded_file.size/1024:.1f} KB</div>
                <div class="info-key">File size</div>
            </div>
            <div class="info-cell" style="grid-column:1/-1">
                <div class="info-val">224 × 224 × 3</div>
                <div class="info-key">Model input shape</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)

    # ── PREPROCESS ───────────────────────────────────────────────────────────
    img_resized = img.resize(IMG_SIZE)
    img_array   = image.img_to_array(img_resized)
    img_array   = np.expand_dims(img_array, axis=0)
    img_array   = preprocess_input(img_array)

    # ── BUTTON ───────────────────────────────────────────────────────────────
    run = st.button("🔬  ANALYSE CELL")

    if run:
        with st.spinner("Neural Network กำลังสแกน…"):
            prediction = float(model.predict(img_array)[0][0])

        confidence = prediction if prediction > 0.5 else 1 - prediction
        conf_pct   = confidence * 100
        is_safe    = prediction > 0.5

        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

        if is_safe:
            st.balloons()
            bar_html  = f'<div class="bar-fill-safe" style="width:{conf_pct:.1f}%"></div>'
            box_cls   = "result-safe"
            icon      = "✅"
            title_th  = "ไม่ติดเชื้อ"
            title_en  = "UNINFECTED · NORMAL CELL"
            pct_color = "#00d2b4"
        else:
            bar_html  = f'<div class="bar-fill-danger" style="width:{conf_pct:.1f}%"></div>'
            box_cls   = "result-danger"
            icon      = "⚠️"
            title_th  = "ตรวจพบเชื้อ"
            title_en  = "INFECTED · PLASMODIUM DETECTED"
            pct_color = "#ff3d6b"

        st.markdown(f"""
        <div class="result-box {box_cls}">
            <div class="result-icon">{icon}</div>
            <div class="result-title" style="color:{pct_color}">{title_th}</div>
            <div class="result-eng">{title_en}</div>
            <div class="conf-pct" style="color:{pct_color}">
                {conf_pct:.1f}<span style="font-size:1.1rem;opacity:.55">%</span>
            </div>
            <div style="font-size:.7rem;color:var(--muted);font-family:'Space Mono',monospace;
                        margin:.25rem 0 .8rem;letter-spacing:.1em">CONFIDENCE</div>
            <div class="bar-track">{bar_html}</div>
            <div style="font-size:.68rem;color:var(--muted);font-family:'Space Mono',monospace;
                        letter-spacing:.08em;margin-top:.6rem">
                sigmoid&nbsp;=&nbsp;{prediction:.5f}&emsp;·&emsp;threshold&nbsp;=&nbsp;0.50000
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

        with st.expander("🔢  Raw model output"):
            st.code(f"""sigmoid output  = {prediction:.8f}
threshold       = 0.50000000
predicted class = {"Uninfected (1)" if is_safe else "Infected (0)"}
confidence      = {conf_pct:.4f} %
backbone        = MobileNetV2  (ImageNet pretrained)
input shape     = (1, 224, 224, 3)
""", language="text")

# ─── FOOTER ────────────────────────────────────────────────────────────────────

