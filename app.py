import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import threading
import queue
import time
import os

# Set Streamlit page configuration
st.set_page_config(page_title="🛡️ Smart Warehouse Surveillance", layout="wide")

st.title("📦 Smart Warehouse Surveillance System")
st.markdown("""
This app performs **real-time detection** of:
- Gunny Bags & Boxes
- Vehicles
- Faces

Upload a video/image file or select your webcam as the input source.
""")

# Load models once (cached)
@st.cache_resource
def load_models():
    yolo_model = YOLO("weights/yolov8n.pt")  # Ensure the path is correct
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return yolo_model, face_cascade

yolo_model, face_cascade = load_models()

# Queue to pass frames safely between thread and main Streamlit
frame_queue = queue.Queue(maxsize=1)

# Detection counts
count_states = {
    "gunny_bags": 0,
    "boxes": 0,
    "vehicles": 0,
    "faces": 0,
}

# Progress indicator
progress_bar = st.sidebar.progress(0)
progress_text = st.sidebar.empty()

def draw_gunny_box(frame, results):
    gunny_count = 0
    box_count = 0
    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            class_name = yolo_model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            # Gunny bags classes (backpack, suitcase)
            if class_name in ["backpack", "suitcase"]:
                label = "Gunny Bag"
                color = (0, 255, 0)
                gunny_count += 1
            # Boxes class (if available)
            elif class_name == "box":
                label = "Box"
                color = (255, 255, 0)
                box_count += 1
            else:
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame, gunny_count, box_count

def draw_vehicles(frame, results):
    vehicle_classes = ["car", "bus", "truck", "motorbike"]
    vehicle_count = 0
    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            class_name = yolo_model.names[cls]
            if class_name not in vehicle_classes:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            color = (0, 0, 255)
            label = class_name.capitalize()
            vehicle_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame, vehicle_count

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
    return frame, len(faces)

def detection_worker(source, total_frames):
    cap = cv2.VideoCapture(source)
    processed_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to a smaller size to improve speed
        frame = cv2.resize(frame, (640, 480))

        # Run YOLO detection
        results = yolo_model(frame, imgsz=640, conf=0.3, verbose=False)

        # Process detections for gunny bags and boxes
        frame, gunny_count, box_count = draw_gunny_box(frame, results)
        count_states["gunny_bags"] = gunny_count
        count_states["boxes"] = box_count

        # Process detections for vehicles
        frame, vehicle_count = draw_vehicles(frame, results)
        count_states["vehicles"] = vehicle_count

        # Detect faces
        frame, face_count = detect_faces(frame)
        count_states["faces"] = face_count

        # Convert BGR to RGB for Streamlit
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Put the latest frame in queue, replacing old if exists
        if not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(rgb_frame)

        # Update progress
        processed_frames += 1
        if total_frames > 0:
            progress = int((processed_frames / total_frames) * 100)
            progress_bar.progress(min(progress, 100))
            progress_text.text(f"Processing... {min(progress, 100)}%")
        else:
            progress_bar.progress(0)
            progress_text.text("Processing...")

        # Sleep to control frame rate (~30 FPS)
        time.sleep(0.03)

    cap.release()
    progress_bar.progress(100)
    progress_text.text("Processing complete.")

# Sidebar input
st.sidebar.header("Select Input Source")
input_type = st.sidebar.radio("Choose input source:", ["Webcam", "Upload Video/Image"])

source = None
total_frames = 0
if input_type == "Webcam":
    cam_index = st.sidebar.number_input("Camera Index", 0, 10, 0)
    source = cam_index
else:
    uploaded_file = st.sidebar.file_uploader("Upload Video/Image", type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        source = tfile.name
        # Estimate total frames for progress bar
        cap = cv2.VideoCapture(source)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

start = st.sidebar.button("Start Detection")

if start:
    if source is None:
        st.sidebar.error("Please select or upload a valid input source.")
    else:
        if "det_thread" not in st.session_state or not st.session_state.det_thread.is_alive():
            det_thread = threading.Thread(target=detection_worker, args=(source, total_frames), daemon=True)
            det_thread.start()
            st.session_state.det_thread = det_thread
            st.success("Detection started!")

# Main UI showing video and live counts
st.subheader("Live Surveillance Feed")
video_placeholder = st.empty()

st.sidebar.subheader("Live Counts")
gunny_count_text = st.sidebar.empty()
box_count_text = st.sidebar.empty()
vehicle_count_text = st.sidebar.empty()
face_count_text = st.sidebar.empty()

def update_counts():
    gunny_count_text.markdown(f"**Gunny Bags:** {count_states['gunny_bags']}")
    box_count_text.markdown(f"**Boxes:** {count_states['boxes']}")
    vehicle_count_text.markdown(f"**Vehicles:** {count_states['vehicles']}")
    face_count_text.markdown(f"**Faces:** {count_states['faces']}")

# Main loop to display frames
while True:
    if not frame_queue.empty():
        frame = frame_queue.get()
        video_placeholder.image(frame)
        update_counts()
    time.sleep(0.03)
