import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os
import pandas as pd
from collections import Counter
import cv2

# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(page_title="Smart Vehicle Detection System", layout="wide")

st.title("🚗 Smart Multi-Vehicle Detection & Traffic Analysis System")

# -----------------------------------
# Load Model
# -----------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Keep best.pt in same folder

model = load_model()

# -----------------------------------
# Traffic Density Logic
# -----------------------------------
def get_density_level(total):
    if total <= 5:
        return "🟢 Low Traffic"
    elif total <= 15:
        return "🟡 Medium Traffic"
    else:
        return "🔴 High Traffic"

# -----------------------------------
# Upload Option
# -----------------------------------
option = st.radio("Select Input Type:", ["Image", "Video"])

# ===================================
# IMAGE DETECTION
# ===================================
if option == "Image":

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        results = model.predict(source=temp_path, conf=0.4)
        result = results[0]

        boxes = result.boxes
        class_ids = boxes.cls.cpu().numpy().astype(int)
        class_names = result.names
        confidences = boxes.conf.cpu().numpy()

        counts = Counter(class_ids)
        total = len(class_ids)

        # Annotated Image
        annotated = result.plot()
        st.image(annotated, caption="Detected Vehicles", use_column_width=True)

        # Traffic Density
        density = get_density_level(total)
        st.subheader("🚦 Traffic Density Level")
        st.write(f"### {density}")

        # Summary
        st.subheader("📊 Detection Summary")
        st.write(f"### Total Vehicles: {total}")

        for cid, count in counts.items():
            st.write(f"**{class_names[cid]}:** {count}")

        # Create Downloadable CSV
        data = []
        for cid, conf in zip(class_ids, confidences):
            data.append({
                "Vehicle Type": class_names[cid],
                "Confidence": round(float(conf), 3)
            })

        df = pd.DataFrame(data)

        st.subheader("📥 Download Detection Results")
        st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False),
            file_name="vehicle_detection_results.csv",
            mime="text/csv"
        )

        os.remove(temp_path)

# ===================================
# VIDEO DETECTION
# ===================================
if option == "Video":

    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        st.video(uploaded_video)

        if st.button("Process Video"):

            results = model.predict(
                source=tfile.name,
                save=True,
                conf=0.4
            )

            st.success("✅ Video Processed Successfully!")
            st.write("Processed video saved in project folder (runs/detect/predict)")
