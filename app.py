import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import threading
import queue
import time
import os

st.set_page_config(page_title="🛡️ Smart Warehouse Surveillance", layout="wide")

st.title("📦 Smart Warehouse Surveillance System")
st.markdown("""
This app performs **real-time detection** of:
- Gunny Bags & Boxes
- Vehicles
- Faces

Upload a video/image file or select your webcam as the single input source.
""")

# Load models once
@st.cache_resource
def load_models():
    yolo_model = YOLO("yolov8n.pt")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return yolo_model, face_cascade

yolo_model, face_cascade = load_models()

frame_queue = queue.Queue(maxsize=1)
count_states = {
    "gunny_bags": 0,
    "boxes": 0,
    "vehicles": 0,
    "faces": 0,
}

frame_counter = 0
video_total_frames = 0

def draw_gunny_box(frame, results):
    gunny_count, box_count = 0, 0
    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            class_name = yolo_model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            if class_name in ["backpack", "suitcase"]:
                label, color = "Gunny Bag", (0, 255, 0)
                gunny_count += 1
            elif class_name == "box":
                label, color = "Box", (255, 255, 0)
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
            label, color = class_name.capitalize(), (0, 0, 255)
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

def detection_worker(source, is_webcam, progress_bar, status_text):
    global frame_counter, video_total_frames
    cap = cv2.VideoCapture(source)
    frame_counter = 0
    if not is_webcam:
        video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        frame = cv2.resize(frame, (640, 480))
        results = yolo_model(frame, imgsz=640, conf=0.3, verbose=False)

        status_text.text("Detecting gunny bags and boxes...")
        frame, gunny_count, box_count = draw_gunny_box(frame, results)
        count_states["gunny_bags"] = gunny_count
        count_states["boxes"] = box_count

        status_text.text("Detecting vehicles...")
        frame, vehicle_count = draw_vehicles(frame, results)
        count_states["vehicles"] = vehicle_count

        status_text.text("Detecting faces...")
        frame, face_count = detect_faces(frame)
        count_states["faces"] = face_count

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(rgb_frame)

        if not is_webcam and video_total_frames > 0:
            progress_bar.progress(min(frame_counter / video_total_frames, 1.0))

        time.sleep(0.03)

    cap.release()
    status_text.text("Detection completed.")

# Sidebar input
st.sidebar.header("Select Input Source")
input_type = st.sidebar.radio("Choose input source:", ["Webcam", "Upload Video/Image"])

source = None
is_webcam = False
if input_type == "Webcam":
    is_webcam = True
    cam_index = st.sidebar.number_input("Camera Index", 0, 10, 0)
    source = cam_index
else:
    uploaded_file = st.sidebar.file_uploader("Upload Video/Image", type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        source = tfile.name

start = st.sidebar.button("Start Detection")

# Display UI
st.subheader("Live Surveillance Feed")
video_placeholder = st.empty()

st.sidebar.subheader("Live Counts")
gunny_count_text = st.sidebar.empty()
box_count_text = st.sidebar.empty()
vehicle_count_text = st.sidebar.empty()
face_count_text = st.sidebar.empty()

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

def update_counts():
    gunny_count_text.markdown(f"**Gunny Bags:** {count_states['gunny_bags']}")
    box_count_text.markdown(f"**Boxes:** {count_states['boxes']}")
    vehicle_count_text.markdown(f"**Vehicles:** {count_states['vehicles']}")
    face_count_text.markdown(f"**Faces:** {count_states['faces']}")

if start:
    if source is None:
        st.sidebar.error("Please select or upload a valid input source.")
    else:
        if "det_thread" not in st.session_state or not st.session_state.det_thread.is_alive():
            det_thread = threading.Thread(
                target=detection_worker,
                args=(source, is_webcam, progress_bar, status_text),
                daemon=True
            )
            det_thread.start()
            st.session_state.det_thread = det_thread
            st.success("Detection started!")

# Display latest frame
if "frame_display" not in st.session_state:
    st.session_state.frame_display = video_placeholder

# Real-time UI update
frame_display = st.session_state.frame_display
if not frame_queue.empty():
    frame = frame_queue.get()
    frame_display.image(frame, channels="RGB")
    update_counts()
