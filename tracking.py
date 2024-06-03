import cv2
import os
from ultralytics import YOLO, solutions

# Define output directory (create it if it doesn't exist)
output_dir = "webcam_frames"
os.makedirs(output_dir, exist_ok=True)

# Initialize video capture object 
input_filename = "HIGH_RES.mp4"
# input_filename = "LOW_RES.mp4"
cap = cv2.VideoCapture(input_filename)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output video codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_filename = os.path.join(output_dir, "output.mp4")
out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

# Run Detection in image
model = YOLO('yolov8m.pt')
names = model.model.names

line_pts = [(0, 800), (2688, 800)]

# Init speed-estimation obj
speed_obj = solutions.SpeedEstimator(
    reg_pts=line_pts,
    names=names,
    view_img=True,
)

# Define frame counter
frame_count = 0

while True:
    # Read a frame from the video
    success, frame = cap.read()

    # Check if frame is read correctly
    if not success:
        print("Error: Unable to capture frame")
        break

    # Detect objects in the frame
    result = model.track(frame, persist=True, show=False)[0]

    frame = speed_obj.estimate_speed(frame, result)

    # Loop through the detected objects
    for box in result.boxes:
        # Extract the bounding box coordinates and class ID
        class_id = result.names[box.cls[0].item()]

        # Check if the detected object is a person
        if class_id == 'person':
            conf = box.conf[0].item()
            cords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [round(x) for x in cords]
            print(class_id, x1, y1)

            # Draw a bounding box around the person
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)
    frame_count += 1

    # Display the frame with bounding boxes
    cv2.imshow('Object Detection', frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release capture, close all windows, and release the VideoWriter object
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Saved {frame_count} frames to {output_filename}")