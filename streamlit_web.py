import streamlit as st
import cv2
import tempfile
import threading
import subprocess
import sys
import os

st.set_page_config(page_title="🛡️ Smart Warehouse Surveillance", layout="wide")

st.title("📦 Smart Warehouse Surveillance Dashboard")
st.markdown("""
This system monitors:
- ✅ Gunny Bags & Boxes
- 🚗 Vehicles
- 😊 Faces

Choose your video source and press **Start Surveillance**.
""")

# Select input source type
source_type = st.radio("Select Input Source:", ("Webcam", "Upload Video/Image"))

if source_type == "Webcam":
    cam_index = st.number_input("Select Camera Index (usually 0 or 1):", min_value=0, max_value=10, value=0, step=1)
    video_path = cam_index  # integer for OpenCV

else:
    uploaded_file = st.file_uploader("Upload an Image or Video file", type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"])
    video_path = None
    if uploaded_file is not None:
        # Save to temp file for OpenCV
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
    else:
        st.warning("Please upload a video or image file.")

# Utility: Run detection script in thread with parameter
def run_script_with_source(script_name, source):
    subprocess.run([sys.executable, script_name, str(source)])

if st.button("▶ Start Full Surveillance"):
    if video_path is None:
        st.error("No valid video source selected!")
    else:
        st.success("Starting surveillance. Please check OpenCV windows.")
        # Run all modules in parallel threads
        threading.Thread(target=run_script_with_source, args=("gunny_box_detector.py", video_path), daemon=True).start()
        threading.Thread(target=run_script_with_source, args=("vehicle_detector.py", video_path), daemon=True).start()
        threading.Thread(target=run_script_with_source, args=("face_detector.py", video_path), daemon=True).start()
