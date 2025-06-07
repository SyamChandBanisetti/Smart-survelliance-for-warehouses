import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # COCO pre-trained

def detect_vehicles(video_source=0):
    cap = cv2.VideoCapture(video_source)
    vehicle_classes = ["car", "bus", "truck", "motorbike"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        vehicle_count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls)
                class_name = model.names[cls]

                if class_name in vehicle_classes:
                    vehicle_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, class_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(frame, f"Vehicles: {vehicle_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Vehicle Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
