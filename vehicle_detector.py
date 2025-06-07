import cv2
from ultralytics import YOLO

# Load YOLOv8 model (default COCO model recognizes vehicle classes well)
model = YOLO("yolov8n.pt")  # Replace with fine-tuned model if needed

# Define vehicle-related classes as per COCO
vehicle_classes = ["car", "truck", "bus", "motorbike", "bicycle", "van"]  # you can modify

def detect_vehicles(video_source=0):
    cap = cv2.VideoCapture(video_source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        vehicle_count = 0

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                class_name = model.names[cls_id]

                if class_name in vehicle_classes:
                    vehicle_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"{class_name}"
                    color = (0, 0, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"Vehicles Detected: {vehicle_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Vehicle Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_vehicles()
