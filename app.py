import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import threading
import queue
import time

st.set_page_config(page_title="🛡️ Smart Warehouse Surveillance", layout="wide")

st.title("📦 Smart Warehouse Surveillance System")

st.markdown("""
This app performs **real-time detection** of:
- Gunny Bags & Boxes
- Vehicles
- Faces

Select source(s) in the sidebar, and watch live detections below.
""")

# Load models once with cache
@st.cache_resource
def load_models():
    yolo_model = YOLO("yolov8n.pt")  # lightweight pretrained COCO model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return yolo_model, face_cascade

yolo_model, face_cascade = load_models()

# Queues for thread-safe frame passing
frame_queues = {
    "gunny_box": queue.Queue(maxsize=1),
    "vehicle": queue.Queue(maxsize=1),
    "face": queue.Queue(maxsize=1),
}

count_states = {
    "gunny_bags": 0,
    "boxes": 0,
    "vehicles": 0,
    "faces": 0,
}

# Function to draw bounding boxes for gunny bags and boxes
def draw_gunny_box(frame, detections):
    gunny_count = 0
    box_count = 0
    for r in detections:
        for box in r.boxes:
            cls = int(box.cls)
            class_name = yolo_model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            # Define "gunny bags" as backpack, suitcase; boxes as 'box' (or suitcase)
            if class_name in ["backpack", "suitcase"]:
                label = "Gunny Bag"
                color = (0, 255, 0)
                gunny_count += 1
            elif class_name == "box":
                label = "Box"
                color = (255, 255, 0)
                box_count += 1
            else:
                continue  # ignore other classes
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame, gunny_count, box_count

# Function to draw bounding boxes for vehicles
def draw_vehicles(frame, detections):
    vehicle_classes = ["car", "bus", "truck", "motorbike"]
    vehicle_count = 0
    for r in detections:
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

# Function to detect and draw faces
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
    return frame, len(faces)

# Worker threads for detection tasks
def detection_worker(name, source, queue_out):
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        if name == "gunny_box":
            results = yolo_model(frame)
            frame, gunny_count, box_count = draw_gunny_box(frame, results)
            count_states["gunny_bags"] = gunny_count
            count_states["boxes"] = box_count
        elif name == "vehicle":
            results = yolo_model(frame)
            frame, vehicle_count = draw_vehicles(frame, results)
            count_states["vehicles"] = vehicle_count
        elif name == "face":
            frame, face_count = detect_faces(frame)
            count_states["faces"] = face_count
        else:
            # Should never happen
            pass

        # Convert BGR to RGB for Streamlit
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Put frame in queue, drop old frame if exists
        if not queue_out.empty():
            try:
                queue_out.get_nowait()
            except queue.Empty:
                pass
        queue_out.put(rgb_frame)
        time.sleep(0.03)  # ~30 FPS

    cap.release()

# Sidebar UI for inputs
st.sidebar.header("Input Sources & Controls")

def get_source_input(key_prefix):
    source_type = st.sidebar.radio(f"{key_prefix} Source", ("Webcam", "Upload Video/Image"), key=f"{key_prefix}_src")
    if source_type == "Webcam":
        cam_index = st.sidebar.number_input(f"{key_prefix} Camera Index", 0, 10, 0, key=f"{key_prefix}_cam")
        return cam_index
    else:
        uploaded_file = st.sidebar.file_uploader(f"Upload {key_prefix} Video/Image", 
                                                 type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"],
                                                 key=f"{key_prefix}_upload")
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            return tfile.name
        else:
            return None

gunny_box_source = get_source_input("Gunny & Boxes")
vehicle_source = get_source_input("Vehicle")
face_source = get_source_input("Face")

# Start buttons to trigger detection threads
start_gunny = st.sidebar.button("Start Gunny & Boxes Detection")
start_vehicle = st.sidebar.button("Start Vehicle Detection")
start_face = st.sidebar.button("Start Face Detection")

# Placeholders for video display
gunny_box_placeholder = st.empty()
vehicle_placeholder = st.empty()
face_placeholder = st.empty()

# Track thread states to prevent multiple starts
threads = {}

def start_detection_thread(name, source):
    if name in threads and threads[name].is_alive():
        return  # already running
    t = threading.Thread(target=detection_worker, args=(name, source, frame_queues[name]), daemon=True)
    t.start()
    threads[name] = t

if start_gunny and gunny_box_source is not None:
    start_detection_thread("gunny_box", gunny_box_source)
elif start_gunny:
    st.sidebar.error("Gunny & Boxes source not selected!")

if start_vehicle and vehicle_source is not None:
    start_detection_thread("vehicle", vehicle_source)
elif start_vehicle:
    st.sidebar.error("Vehicle source not selected!")

if start_face and face_source is not None:
    start_detection_thread("face", face_source)
elif start_face:
    st.sidebar.error("Face source not selected!")

# Main UI columns to display streams and counts
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Gunny Bags & Boxes")
    if "gunny_box" in threads and threads["gunny_box"].is_alive():
        if not frame_queues["gunny_box"].empty():
            frame = frame_queues["gunny_box"].get()
            st.image(frame)
        st.write(f"Gunny Bags detected: **{count_states['gunny_bags']}**")
        st.write(f"Boxes detected: **{count_states['boxes']}**")
    else:
        st.write("Not running")

with col2:
    st.subheader("Vehicle Detection")
    if "vehicle" in threads and threads["vehicle"].is_alive():
        if not frame_queues["vehicle"].empty():
            frame = frame_queues["vehicle"].get()
            st.image(frame)
        st.write(f"Vehicles detected: **{count_states['vehicles']}**")
    else:
        st.write("Not running")

with col3:
    st.subheader("Face Detection")
    if "face" in threads and threads["face"].is_alive():
        if not frame_queues["face"].empty():
            frame = frame_queues["face"].get()
            st.image(frame)
        st.write(f"Faces detected: **{count_states['faces']}**")
    else:
        st.write("Not running")

# Footer note
st.markdown("---")
st.markdown("Made with ❤️ by Syam Chand - Smart Warehouse Surveillance Demo")

