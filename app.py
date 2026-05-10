import st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# --- CONFIGURATION ---
MODEL_PATH = "best_model (2).keras"
IMG_SIZE = (224, 224)
SAMPLE_DIR = "samples"  # ชื่อโฟลเดอร์ที่มึงเก็บรูปไว้บน GitHub

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="MalariaScope · CMU",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── MASTER CSS (เหมือนเดิมเป๊ะ กูย่อไว้) ────────────────────────────────────────
st.markdown("""
<style>
/* CSS ทั้งหมดที่มึงมีใส่ไว้ตรงนี้เหมือนเดิม */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Outfit:wght@300;400;500&display=swap');
:root { --ink: #04080f; --surface: #080e1a; --cyan: #00d2b4; --blue: #0a84ff; --red: #ff3d6b; --muted: #4d7a99; --txt: #d8eeff; --border: rgba(0,210,180,0.13); --border2: rgba(0,210,180,0.28); --r: 18px; }
#MainMenu, footer, header { visibility: hidden; }
html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; background-color: var(--ink) !important; color: var(--txt) !important; }
.grid-bg { position: fixed; inset: 0; z-index: 0; background-image: linear-gradient(rgba(0,210,180,.04) 1px, transparent 1px), linear-gradient(90deg, rgba(0,210,180,.04) 1px, transparent 1px); background-size: 48px 48px; pointer-events: none; }
.hero-wrap { position: relative; padding: 3.5rem 1rem 2.5rem; text-align: center; }
.hero-title { font-family: 'Syne', sans-serif; font-size: clamp(2.4rem, 6vw, 3.6rem); font-weight: 800; background: linear-gradient(160deg, #ffffff 20%, var(--cyan) 60%, var(--blue) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.section-label { font-family: 'Space Mono', monospace; font-size: .65rem; letter-spacing: .18em; color: var(--muted); text-transform: uppercase; margin-bottom: 1rem; display: flex; align-items: center; gap: .5rem; }
.section-label::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, var(--border2), transparent); }
.info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .7rem; }
.info-cell { background: rgba(0,210,180,.05); border: 1px solid var(--border); border-radius: 10px; padding: .75rem 1rem; }
.info-val { font-family: 'Space Mono', monospace; font-size: .95rem; color: var(--txt); font-weight: 700; }
.info-key { font-size: .68rem; color: var(--muted); text-transform: uppercase; }
[data-testid="stButton"] > button { width: 100% !important; background: linear-gradient(135deg, var(--cyan) 0%, var(--blue) 100%) !important; color: #040c18 !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; border-radius: 12px !important; padding: .9rem 2rem !important; text-transform: uppercase !important; }
.result-box { border-radius: var(--r); padding: 2.4rem 2rem; text-align: center; position: relative; }
.result-safe { background: rgba(0,210,180,.08); border: 1px solid rgba(0,210,180,.32); }
.result-danger { background: rgba(255,61,107,.08); border: 1px solid rgba(255,61,107,.34); }
.conf-pct { font-family: 'Space Mono', monospace; font-size: 2.6rem; font-weight: 700; }
.bar-track { height: 6px; background: rgba(255,255,255,.07); border-radius: 999px; overflow: hidden; margin: .7rem auto; max-width: 340px; }
.bar-fill-safe { height:100%; background: linear-gradient(90deg,var(--cyan),var(--blue)); }
.bar-fill-danger { height:100%; background: linear-gradient(90deg,var(--red),var(--amber)); }
</style>
<div class="grid-bg"></div>
""", unsafe_allow_html=True)

# ─── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <p class="hero-eyebrow">🧬 &nbsp; DATASCI CMU</p>
    <h1 class="hero-title">Malaria<br>Scope</h1>
    <p class="hero-desc">วิเคราะห์เซลล์เม็ดเลือดแดงด้วย MobileNetV2<br>เลือกภาพตัวอย่างจากคลังเพื่อเริ่มต้น</p>
</div>
""", unsafe_allow_html=True)

# ─── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_my_model()
except Exception:
    st.error(f"⚠️ ไม่พบไฟล์ `{MODEL_PATH}`")
    st.stop()

# ─── SAMPLE IMAGE SELECTION ─────────────────────────────────────────────────────
st.markdown('<div class="section-label">🔬 &nbsp;เลือกภาพตัวอย่างจากคลัง (SAMPLES)</div>', unsafe_allow_html=True)

if not os.path.exists(SAMPLE_DIR):
    st.warning(f" ไม่พบโฟลเดอร์ `{SAMPLE_DIR}` กรุณาอัปโหลดไว้ใน GitHub ด้วย")
    st.stop()

sample_files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not sample_files:
    st.error(" ⚠️ ไม่มีไฟล์รูปภาพในโฟลเดอร์ samples")
    st.stop()

# Dropdown สำหรับเลือกรูป
selected_filename = st.selectbox("เลือกภาพที่ต้องการวิเคราะห์:", sample_files)

if selected_filename:
    img_path = os.path.join(SAMPLE_DIR, selected_filename)
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    file_size = os.path.getsize(img_path) / 1024

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
                <div class="info-val">{selected_filename}</div>
                <div class="info-key">Filename</div>
            </div>
            <div class="info-cell">
                <div class="info-val">{w}×{h}</div>
                <div class="info-key">Resolution</div>
            </div>
            <div class="info-cell">
                <div class="info-val">{file_size:.1f} KB</div>
                <div class="info-key">File size</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── PREPROCESS & PREDICT ─────────────────────────────────────────────────
    img_resized = img.resize(IMG_SIZE)
    img_array   = image.img_to_array(img_resized)
    img_array   = np.expand_dims(img_array, axis=0)
    img_array   = preprocess_input(img_array)

    run = st.button("🔬 ANALYSE SELECTED CELL")

    if run:
        with st.spinner("Neural Network กำลังสแกน…"):
            prediction = float(model.predict(img_array)[0][0])

        confidence = prediction if prediction > 0.5 else 1 - prediction
        conf_pct   = confidence * 100
        is_safe    = prediction > 0.5

        if is_safe:
            st.balloons()
            bar_html  = f'<div class="bar-fill-safe" style="width:{conf_pct:.1f}%"></div>'
            box_cls   = "result-safe"
            icon, title_th, title_en, pct_color = "✅", "ไม่ติดเชื้อ", "UNINFECTED · NORMAL", "#00d2b4"
        else:
            bar_html  = f'<div class="bar-fill-danger" style="width:{conf_pct:.1f}%"></div>'
            box_cls   = "result-danger"
            icon, title_th, title_en, pct_color = "⚠️", "ตรวจพบเชื้อ", "INFECTED · MALARIA", "#ff3d6b"

        st.markdown(f"""
        <div class="result-box {box_cls}">
            <div class="result-icon">{icon}</div>
            <div class="result-title" style="color:{pct_color}">{title_th}</div>
            <div class="result-eng">{title_en}</div>
            <div class="conf-pct" style="color:{pct_color}">{conf_pct:.1f}%</div>
            <div class="bar-track">{bar_html}</div>
            <div style="font-size:.68rem;color:var(--muted);font-family:'Space Mono',monospace;margin-top:.6rem">
                sigmoid&nbsp;=&nbsp;{prediction:.5f}&emsp;·&emsp;threshold&nbsp;=&nbsp;0.50
            </div>
        </div>
        """, unsafe_allow_html=True)
