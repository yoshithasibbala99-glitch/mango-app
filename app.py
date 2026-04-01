import streamlit as st

st.title("Test App")

@st.cache_resource
def load_model():
from ultralytics import YOLO
return YOLO("best.pt")

model = load_model()

st.success("Model Loaded Successfully ✅")
