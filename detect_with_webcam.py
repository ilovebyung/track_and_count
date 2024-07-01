import cv2
from ultralytics import YOLO

# # Define the codec for MP4 video (FourCC code)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')

# # Define output video filename
# output_filename = "output.mp4"

# Open the webcam
cap = cv2.VideoCapture(0)

# # Get frame width and height
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Define video writer object with desired frame size, codec, and FPS (Frames per Second)
# out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))

# print("Recording started. Press 'q' to stop recording.")

# Run Detection in image
model = YOLO('yolov8n.pt')

# Load the exported ONNX model
# model = YOLO("yolov8n.onnx")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Detect objects in the frame
    result = model(frame)[0]

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

    # # Write the frame into the video writer
    # out.write(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and video writer objects
cap.release()
# out.release()
# Close all open windows
cv2.destroyAllWindows()

# print("Recording stopped. Video saved as", output_filename)
