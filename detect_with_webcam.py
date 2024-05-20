import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# Open the webcam
cap = cv2.VideoCapture(0)

# Run Detection in image
model = YOLO('yolov8s.pt')

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Detect objects in the frame
    result = model(frame)[0]

    # # Loop through the detected objects
    # for result in results.xyxy[0]:
    #     # Extract the bounding box coordinates and class ID
    #     x1, y1, x2, y2, conf, cls = result

    #     # Check if the detected object is a person
    #     if int(cls) == 0:  # 0 is the class ID for 'person' in the COCO dataset
    #         # Draw a bounding box around the person
    #         cv2.rectangle(frame, (int(x1), int(y1)),
    #                       (int(x2), int(y2)), (0, 255, 0), 2)

    # Loop through the detected objects
    for box in result.boxes:
        # Extract the bounding box coordinates and class ID
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        x1, y1, x2, y2 = [round(x) for x in cords]
        print(class_id, x1, y1)

        # Check if the detected object is a person
        if class_id == 'person':  # 0 is the class ID for 'person' in the COCO dataset
            # Draw a bounding box around the person
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow('YOLOv8 Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
