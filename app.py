import streamlit as st
from PIL import Image, ImageDraw
import tempfile
import os
import time

st.set_page_config(
    page_title="Mango Tree Counter",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(160deg, #071a09 0%, #0d2b12 60%, #0f3016 100%); min-height: 100vh; }
header[data-testid="stHeader"] { background: transparent; }
.hero { text-align: center; padding: 3rem 1rem 1.5rem; }
.hero-badge { display: inline-block; background: rgba(134,239,92,0.1); border: 1px solid rgba(134,239,92,0.3); color: #86ef5c; font-size: 0.7rem; letter-spacing: 0.18em; text-transform: uppercase; padding: 4px 14px; border-radius: 999px; margin-bottom: 14px; }
.hero h1 { font-family: 'Syne', sans-serif; font-size: clamp(2rem, 5vw, 3.4rem); font-weight: 800; color: #f0fdf0; line-height: 1.1; margin: 0 0 10px; }
.hero h1 span { color: #86ef5c; }
.hero p { color: rgba(210,252,210,0.45); font-size: 1rem; max-width: 480px; margin: 0 auto; }
.steps-row { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; margin: 1.8rem auto; max-width: 860px; }
.step-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(134,239,92,0.12); border-radius: 14px; padding: 14px 18px; flex: 1; min-width: 160px; max-width: 220px; }
.step-num { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 800; color: rgba(134,239,92,0.35); margin-bottom: 4px; }
.step-title { font-size: 0.82rem; font-weight: 500; color: #b6f5b6; margin-bottom: 3px; }
.step-desc { font-size: 0.72rem; color: rgba(210,252,210,0.4); line-height: 1.5; }
.metrics-row { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; margin: 1.2rem auto 1.8rem; max-width: 700px; }
.metric-card { background: rgba(255,255,255,0.05); border: 1px solid rgba(134,239,92,0.18); border-radius: 14px; padding: 14px 24px; text-align: center; min-width: 140px; }
.metric-val { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #86ef5c; line-height: 1; }
.metric-lbl { font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; color: rgba(210,252,210,0.38); margin-top: 4px; }
.upload-wrap { max-width: 660px; margin: 0 auto 2rem; }
[data-testid="stFileUploader"] { background: rgba(255,255,255,0.025) !important; border: 2px dashed rgba(134,239,92,0.28) !important; border-radius: 16px !important; padding: 2rem !important; }
[data-testid="stFileUploader"] label { color: rgba(210,252,210,0.65) !important; }
.divider { border: none; border-top: 1px solid rgba(134,239,92,0.1); margin: 1.6rem 0; }
.rc { background: rgba(255,255,255,0.035); border: 1px solid rgba(134,239,92,0.1); border-radius: 18px; overflow: hidden; margin-bottom: 1.4rem; }
.rc-head { padding: 10px 16px; display: flex; align-items: center; justify-content: space-between; background: rgba(134,239,92,0.05); border-bottom: 1px solid rgba(134,239,92,0.08); }
.rc-name { font-family: 'Syne', sans-serif; font-weight: 700; color: #d1fad1; font-size: 0.88rem; }
.count-pill { background: #86ef5c; color: #071a09; font-family: 'Syne', sans-serif; font-weight: 800; font-size: 0.78rem; padding: 3px 12px; border-radius: 999px; }
.img-lbl { font-size: 0.65rem; letter-spacing: 0.12em; text-transform: uppercase; color: rgba(210,252,210,0.35); margin-bottom: 5px; }
.total-banner { background: linear-gradient(90deg, rgba(134,239,92,0.14), rgba(134,239,92,0.04)); border: 1px solid rgba(134,239,92,0.3); border-radius: 18px; padding: 1.6rem 2rem; text-align: center; margin: 0.5rem auto 2rem; max-width: 420px; }
.total-num { font-family: 'Syne', sans-serif; font-size: 3.2rem; font-weight: 800; color: #86ef5c; line-height: 1; }
.total-lbl { color: rgba(210,252,210,0.55); font-size: 0.9rem; margin-top: 6px; }
div[data-testid="stImage"] img { border-radius: 10px; width: 100%; }
.stProgress > div > div { background: #86ef5c !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model — import torch directly, skip cv2 entirely ─────────────────────
@st.cache_resource
def load_model():
    import torch
    from ultralytics import YOLO
    model = YOLO("best.pt")
    return model

def draw_boxes(image: Image.Image, boxes) -> Image.Image:
    """Draw bounding boxes using PIL only — zero cv2 needed."""
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        draw.rectangle([x1, y1, x2, y2], outline="#00ff00", width=3)
        draw.rectangle([x1, y1, x1 + 55, y1 + 16], fill="#00ff00")
        draw.text((x1 + 3, y1 + 1), "Tree", fill="#000000")
    return img

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">YOLOv8 · Roboflow · Ultralytics</div>
  <h1>Mango Tree<br><span>Detection & Counter</span></h1>
  <p>Upload one or more field or aerial images — the AI model will detect and count every mango tree automatically.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="steps-row">
  <div class="step-card"><div class="step-num">01</div><div class="step-title">📸 150 Photos Captured</div><div class="step-desc">Field images of mango trees collected and organised.</div></div>
  <div class="step-card"><div class="step-num">02</div><div class="step-title">🏷️ Annotated on Roboflow</div><div class="step-desc">Drew bounding boxes around every tree to create the labelled dataset.</div></div>
  <div class="step-card"><div class="step-num">03</div><div class="step-title">🧠 Trained YOLOv8</div><div class="step-desc">Ran 50 epochs on Google Colab using the annotated dataset.</div></div>
  <div class="step-card"><div class="step-num">04</div><div class="step-title">🌴 Detect & Count</div><div class="step-desc">Upload any new image and the trained model counts trees instantly.</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

st.markdown('<div class="upload-wrap">', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Upload mango tree images (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_files:
    model = load_model()
    results_data = []
    status = st.empty()
    bar = st.progress(0)

    for i, f in enumerate(uploaded_files):
        status.markdown(
            f"<p style='color:rgba(210,252,210,0.5);text-align:center;font-size:0.84rem;'>"
            f"Analysing <b style='color:#86ef5c'>{f.name}</b> ...</p>",
            unsafe_allow_html=True,
        )
        bar.progress(i / len(uploaded_files))
        image = Image.open(f).convert("RGB")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        result = model(tmp_path, verbose=False)[0]
        count = len(result.boxes)
        annotated = draw_boxes(image, result.boxes)

        results_data.append({"name": f.name, "original": image, "detected": annotated, "count": count})
        os.unlink(tmp_path)

    bar.progress(1.0)
    time.sleep(0.3)
    status.empty()
    bar.empty()

    total = sum(r["count"] for r in results_data)
    n_imgs = len(results_data)
    avg = round(total / n_imgs, 1) if n_imgs else 0

    st.markdown(f"""
    <div class="metrics-row">
      <div class="metric-card"><div class="metric-val">{total}</div><div class="metric-lbl">Trees Detected</div></div>
      <div class="metric-card"><div class="metric-val">{n_imgs}</div><div class="metric-lbl">Images Analysed</div></div>
      <div class="metric-card"><div class="metric-val">{avg}</div><div class="metric-lbl">Avg per Image</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("<h3 style='font-family:Syne,sans-serif;color:#d1fad1;font-size:1.1rem;margin-bottom:1rem;'>Detection Results</h3>", unsafe_allow_html=True)

    for r in results_data:
        st.markdown(f"""
        <div class="rc"><div class="rc-head">
          <span class="rc-name">{r['name']}</span>
          <span class="count-pill">🌴 {r['count']} trees</span>
        </div></div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="img-lbl">Original</p>', unsafe_allow_html=True)
            st.image(r["original"], use_column_width=True)
        with col2:
            st.markdown('<p class="img-lbl">Detected</p>', unsafe_allow_html=True)
            st.image(r["detected"], use_column_width=True)
        st.markdown("<div style='margin-bottom:1.2rem'></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="total-banner">
      <div class="total-num">🌴 {total}</div>
      <div class="total-lbl">Total Mango Trees Detected Across All Images</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("<p style='text-align:center;color:rgba(210,252,210,0.28);font-size:0.88rem;margin-top:0.5rem;'>Upload one or more images above to start detection.</p>", unsafe_allow_html=True)
