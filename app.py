# 🌳 MangoAI – Tree Counter (FREE Satellite + KML)

import streamlit as st
from PIL import Image, ImageDraw
import io
import pandas as pd
import requests
import xml.etree.ElementTree as ET

# ── PAGE ───────────────────────────────────────────

st.set_page_config(page_title="MangoAI Tree Counter", layout="wide")
st.title("🌳 MangoAI – Tree Detection & Counter (FREE MODE)")

# ── LOAD MODEL ─────────────────────────────────────

@st.cache_resource
def load_model():
from ultralytics import YOLO
return YOLO("best.pt")

model = load_model()

# ── DRAW BOXES ─────────────────────────────────────

def draw_boxes(image, boxes, conf_threshold=0.35):
img = image.copy().convert("RGB")
draw = ImageDraw.Draw(img)
count = 0

```
for box in boxes:
    conf = float(box.conf[0])
    if conf < conf_threshold:
        continue

    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
    draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
    count += 1

return img, count
```

# ── KML PARSER ─────────────────────────────────────

def extract_coordinates_from_kml(kml_file):
tree = ET.parse(kml_file)
root = tree.getroot()

```
coords = []
for elem in root.iter():
    if 'coordinates' in elem.tag:
        raw = elem.text.strip()
        points = raw.split()

        for p in points:
            lon, lat, *_ = map(float, p.split(','))
            coords.append((lat, lon))

return coords
```

# ── GRID GENERATOR ─────────────────────────────────

def generate_grid(coords, step=0.0005):
lats = [c[0] for c in coords]
lons = [c[1] for c in coords]

```
lat_min, lat_max = min(lats), max(lats)
lon_min, lon_max = min(lons), max(lons)

grid = []
lat = lat_min

while lat < lat_max:
    lon = lon_min
    while lon < lon_max:
        grid.append((lat, lon))
        lon += step
    lat += step

return grid
```

# ── FREE SATELLITE IMAGE (ESRI) ────────────────────

def get_satellite_image(lat, lon):
delta = 0.001  # area size

```
bbox = f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}"

url = f"https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={bbox}&bboxSR=4326&size=640,640&imageSR=4326&format=png"

response = requests.get(url)

if response.status_code != 200:
    return None

return Image.open(io.BytesIO(response.content))
```

# ── UI ─────────────────────────────────────────────

mode = st.radio("Select Input Mode", ["📷 Upload Images", "📍 Upload KML (Auto Detect)"])
confidence = st.slider("Detection Confidence", 0.1, 0.9, 0.35, 0.05)

# ── IMAGE MODE ─────────────────────────────────────

if mode == "📷 Upload Images":
uploaded_files = st.file_uploader("Upload Images", type=["jpg","png"], accept_multiple_files=True)

```
if uploaded_files:
    total = 0
    results = []

    for f in uploaded_files:
        image = Image.open(f)
        result = model(image)[0]

        annotated, count = draw_boxes(image, result.boxes, confidence)

        st.image(annotated, caption=f"{f.name} → {count} trees")
        total += count

        results.append({"Image": f.name, "Trees": count})

    st.success(f"🌳 Total Trees: {total}")

    df = pd.DataFrame(results)
    st.dataframe(df)
```

# ── KML MODE ───────────────────────────────────────

elif mode == "📍 Upload KML (Auto Detect)":
kml_file = st.file_uploader("Upload KML file", type=["kml"])

```
if kml_file:
    st.info("📍 Extracting coordinates...")
    coords = extract_coordinates_from_kml(kml_file)

    st.info("🛰️ Generating grid...")
    grid = generate_grid(coords)

    st.write(f"Total tiles: {len(grid)}")

    progress = st.progress(0)
    total_trees = 0

    for i, (lat, lon) in enumerate(grid):
        img = get_satellite_image(lat, lon)

        if img is None:
            continue

        result = model(img)[0]
        _, count = draw_boxes(img, result.boxes, confidence)

        total_trees += count
        progress.progress(i / len(grid))

    progress.progress(1.0)

    st.success(f"🌳 Total Trees in Area: {total_trees}")
```
