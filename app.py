import streamlit as st
from PIL import Image, ImageDraw
import io
import requests
import xml.etree.ElementTree as ET

st.title("🌳 MangoAI Tree Counter")

# ✅ FIXED FUNCTION

@st.cache_resource
def load_model():
from ultralytics import YOLO
return YOLO("best.pt")

model = load_model()

def draw_boxes(image, boxes):
img = image.copy()
draw = ImageDraw.Draw(img)
count = 0

```
for box in boxes:
    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
    draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
    count += 1

return img, count
```

def get_satellite_image(lat, lon):
delta = 0.001
bbox = f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}"

```
url = f"https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={bbox}&bboxSR=4326&size=640,640&imageSR=4326&format=png"

response = requests.get(url)

if response.status_code != 200:
    return None

return Image.open(io.BytesIO(response.content))
```

st.write("Upload Image")

file = st.file_uploader("Upload", type=["jpg","png"])

if file:
img = Image.open(file)
result = model(img)[0]

```
annotated, count = draw_boxes(img, result.boxes)

st.image(annotated)
st.success(f"Trees: {count}")
```
