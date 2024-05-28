import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import csv

# Load your YOLOv8 model (replace 'path/to/yolov8n.pt' with your model path)
model = YOLO('./trafficdata/data/labels/runs/detect/train8/weights/best.pt')

# Load your Keras classification model (replace 'path/to/classification_model.h5' with your model path)
classification_model = load_model(
    './customModels/model_yabancı_2(1).h5')

# Define the class names for your classification model
class_names = []# Replace with your actual class names

def get_class_names(class_names: list) -> None:
    with open("./trafficdata/labels.csv", "r") as file:
        file_reader = csv.reader(file, delimiter=",")
        for row in file_reader:
            if row[0] != "ClassId":
                class_names.append(row[1])
        file.close()


get_class_names(class_names)


# Set up video capture (0 is the default camera)
video_capture = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not video_capture.isOpened():
    raise IOError("Cannot open webcam")

while True:
    read_success, frame = video_capture.read()
    if not read_success:
        break

    # Perform inference with YOLOv8
    results = model(frame)

    # Parse results
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        for box, score, class_id in zip(boxes, scores, class_ids):
            if score > 0.5:  # You can adjust the confidence threshold
                x1, y1, x2, y2 = map(int, box)
                traffic_sign = frame[y1:y2, x1:x2]  # Crop the traffic sign region

                # Convert to grayscale if the model expects grayscale images
                traffic_sign_gray = cv2.cvtColor(traffic_sign, cv2.COLOR_BGR2GRAY)
                traffic_sign_resized = cv2.resize(traffic_sign_gray, (32, 32))  # Resize to your model input size
                traffic_sign_array = img_to_array(traffic_sign_resized)
                traffic_sign_array = np.expand_dims(traffic_sign_array, axis=0)  # Add batch dimension
                traffic_sign_array = np.expand_dims(traffic_sign_array, axis=-1)  # Add channel dimension for grayscale
                traffic_sign_array /= 255.0  # Normalize if needed

                # Perform classification
                traffic_sign_prediction = classification_model.predict(traffic_sign_array)
                predicted_class = np.argmax(traffic_sign_prediction, axis=1)
                class_name = class_names[predicted_class[0]]

                # Draw bounding box and classification result on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2)

    # Display the frame with detections and classifications
    cv2.imshow('YOLOv8 Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
