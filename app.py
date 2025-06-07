import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

st.set_page_config(page_title="🛡️ Smart Warehouse Surveillance", layout="wide")

st.title("📦 Smart Warehouse Surveillance System")
st.markdown("""
Monitor warehouse with these features:
- **Gunny Bags & Boxes Detection**
- **Vehicle Detection**
- **Face Detection**

Use webcam or upload images/videos to test.
Press **Start** and see real-time detection results below.
""")

@st.cache_resource
def load_models():
    yolo_model = YOLO("yolov8n.pt")  # COCO pretrained
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return yolo_model, face_cascade

yolo_model, face_cascade = load_models()

def draw_boxes_on_frame(frame, detections, classes_to_detect, label_map):
    gunny_count, box_count, vehicle_count = 0, 0, 0
    for r in detections:
        for box in r.boxes:
            cls = int(box.cls)
            class_name = yolo_model.names[cls]
            if class_name not in classes_to_detect:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            if class_name in label_map["gunny"]:
                gunny_count += 1
                label = "Gunny Bag"
                color = (0, 255, 0)
            elif class_name in label_map["box"]:
                box_count += 1
                label = "Box"
                color = (255, 255, 0)
            elif class_name in label_map["vehicle"]:
                vehicle_count += 1
                label = class_name.capitalize()
                color = (0, 0, 255)
            else:
                label = class_name.capitalize()
                color = (255, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame, gunny_count, box_count, vehicle_count

def detect_faces_on_frame(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
    return frame, len(faces)

def process_video_stream(source, mode):
    cap = cv2.VideoCapture(source)
    FRAME_WINDOW = st.image([])
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if mode == "Gunny Bags & Boxes":
            classes_to_detect = ["backpack", "suitcase", "box"]
            label_map = {
                "gunny": ["backpack", "suitcase"],
                "box": ["box"],
                "vehicle": []
            }
            results = yolo_model(frame)
            frame, gunny_count, box_count, _ = draw_boxes_on_frame(frame, results, classes_to_detect, label_map)
            cv2.putText(frame, f"Gunny Bags: {gunny_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Boxes: {box_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        elif mode == "Vehicle Detection":
            vehicle_classes = ["car", "bus", "truck", "motorbike"]
            label_map = {
                "gunny": [],
                "box": [],
                "vehicle": vehicle_classes
            }
            results = yolo_model(frame)
            frame, _, _, vehicle_count = draw_boxes_on_frame(frame, results, vehicle_classes, label_map)
            cv2.putText(frame, f"Vehicles: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        else:  # Face Detection
            frame, face_count = detect_faces_on_frame(frame, face_cascade)
            cv2.putText(frame, f"Faces: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

tab1, tab2, tab3 = st.tabs(["Gunny Bags & Boxes", "Vehicle Detection", "Face Detection"])

with tab1:
    st.header("Gunny Bags & Boxes Detection")
    source_type = st.radio("Input source:", ["Webcam", "Upload Video/Image"])
    video_src = None
    if source_type == "Webcam":
        cam_idx = st.number_input("Select Camera Index:", min_value=0, max_value=10, value=0, step=1)
        video_src = cam_idx
    else:
        uploaded_file = st.file_uploader("Upload image/video file", type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_src = tfile.name
        else:
            st.warning("Upload a file to start detection.")
    if st.button("Start Gunny & Boxes Detection"):
        if video_src is None:
            st.error("Please select webcam or upload a file.")
        else:
            process_video_stream(video_src, "Gunny Bags & Boxes")

with tab2:
    st.header("Vehicle Detection")
    source_type2 = st.radio("Input source:", ["Webcam", "Upload Video/Image"], key="vehicle")
    video_src2 = None
    if source_type2 == "Webcam":
        cam_idx2 = st.number_input("Select Camera Index:", min_value=0, max_value=10, value=0, step=1, key="vehicle_cam")
        video_src2 = cam_idx2
    else:
        uploaded_file2 = st.file_uploader("Upload image/video file", type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"], key="vehicle_upload")
        if uploaded_file2 is not None:
            tfile2 = tempfile.NamedTemporaryFile(delete=False)
            tfile2.write(uploaded_file2.read())
            video_src2 = tfile2.name
        else:
            st.warning("Upload a file to start detection.")
    if st.button("Start Vehicle Detection", key="start_vehicle"):
        if video_src2 is None:
            st.error("Please select webcam or upload a file.")
        else:
            process_video_stream(video_src2, "Vehicle Detection")

with tab3:
    st.header("Face Detection")
    source_type3 = st.radio("Input source:", ["Webcam", "Upload Video/Image"], key="face")
    video_src3 = None
    if source_type3 == "Webcam":
        cam_idx3 = st.number_input("Select Camera Index:", min_value=0, max_value=10, value=0, step=1, key="face_cam")
        video_src3 = cam_idx3
    else:
        uploaded_file3 = st.file_uploader("Upload image/video file", type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"], key="face_upload")
        if uploaded_file3 is not None:
            tfile3 = tempfile.NamedTemporaryFile(delete=False)
            tfile3.write(uploaded_file3.read())
            video_src3 = tfile3.name
        else:
            st.warning("Upload a file to start detection.")
    if st.button("Start Face Detection", key="start_face"):
        if video_src3 is None:
            st.error("Please select webcam or upload a file.")
        else:
            process_video_stream(video_src3, "Face Detection")
