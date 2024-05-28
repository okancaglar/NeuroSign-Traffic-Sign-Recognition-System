import cv2
import torch
from ultralytics import YOLO

# Check if setNumThreads is available
if hasattr(cv2, 'setNumThreads'):
    cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
#/home/marcus-aurelius/Projects/SchoolProjects/MLProjectV1/trafficdata/data/labelqqs/runs/detect/train8/weights/best.pt
# Load your YOLOv8 model (replace 'path/to/best.pt' with your model path)
model = YOLO('./trafficdata/data/labels/runs/detect/train8/weights/best.pt')
#model.predict("/home/marcus-aurelius/Projects/SchoolProjects/MLProjectV1/trafficdata/data/images/train", save=False, conf=0.5)

# Set up video capture (0 is the default camera)
video_capture = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not video_capture.isOpened():
    raise IOError("Cannot open webcam")

while True:
    read_success, captured_frame = video_capture.read()
    if not read_success:
        break

    # Perform inference
    results = model(captured_frame)

    # Parse results
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        # Draw bounding boxes and labels on the frame
        for box, score, class_id in zip(boxes, scores, class_ids):
            if score > 0.5:  # You can adjust the confidence threshold
                x1, y1, x2, y2 = map(int, box)
                class_name = model.names[int(class_id)]
                cv2.rectangle(captured_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(captured_frame, f"{class_name} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow('YOLOv8 Detection', captured_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
