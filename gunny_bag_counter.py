import cv2
from ultralytics import YOLO

# Load YOLOv8 model (can be custom-trained for gunny_bag and box)
model = YOLO("yolov8n.pt")  # Use "best.pt" if you've trained on gunny_bag and box classes

# Define your class labels here (based on training or COCO mapping)
TARGET_CLASSES = ["gunny_bag", "box", "suitcase", "backpack"]  # adjust as per your model

def count_items(video_source=0):
    cap = cv2.VideoCapture(video_source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        gunny_bag_count = 0
        box_count = 0

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                class_name = model.names[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{class_name}"

                if class_name in ["gunny_bag", "suitcase", "backpack"]:
                    gunny_bag_count += 1
                    color = (0, 255, 0)
                elif class_name == "box":
                    box_count += 1
                    color = (255, 255, 0)
                else:
                    continue  # Ignore other classes

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"Gunny Bags: {gunny_bag_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Boxes: {box_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        cv2.imshow("Gunny Bag & Box Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    count_items()
