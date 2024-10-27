import cv2
import torch

# Open the default camera (0 for the first camera)
cap = cv2.VideoCapture(0)

# Load the YOLOv5 model (using a smaller model for better performance)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, device=device)

# Get all COCO class names
coco_names = model.names

frame_skip = 2  # Process every other frame (adjust this value as needed)
delay_frames = 15  # Number of frames to keep the rectangles and labels (adjust as needed)
object_detections = []  # List to store object detections
previous_objects = set()  # Set to store unique objects detected in previous frames

frame_counter = 0

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame from camera")
        break

    frame_counter += 1

    # Process every other frame
    if frame_counter % frame_skip == 0:
        # Resize frame to the model's input resolution
        img = cv2.resize(frame, (640, 480))

        # Perform object detection
        results = model(img)

        # Add new detections to the list
        for pred in results.pred[0]:
            x1, y1, x2, y2 = [int(x) for x in pred[:4]]
            conf = pred[4]
            class_id = int(pred[5])
            class_name = coco_names[class_id]
            object_tuple = (class_name, x1, y1, x2, y2)

            # Check if the object is new or has moved significantly
            if object_tuple not in previous_objects:
                object_detections.append((class_name, conf, x1, y1, x2, y2, delay_frames))
                previous_objects.add(object_tuple)

    # Draw bounding boxes and labels for each object detection
    for detection in object_detections[:]:
        class_name, conf, x1, y1, x2, y2, delay = detection

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

        delay -= 1
        if delay == 0:
            object_detections.remove(detection)
            previous_objects.remove((class_name, x1, y1, x2, y2))

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
print("Program ended.")
