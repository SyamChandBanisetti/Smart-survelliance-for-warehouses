import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Pretrained COCO model

def detect_gunny_and_boxes(video_source=0):
    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        gunny_count, box_count = 0, 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls)
                class_name = model.names[cls]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 255, 0)

                if class_name in ["backpack", "suitcase"]:
                    gunny_count += 1
                    label = "Gunny Bag"
                elif class_name in ["box"]:
                    box_count += 1
                    label = "Box"
                else:
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"Gunny Bags: {gunny_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Boxes: {box_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Gunny Bag & Box Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
