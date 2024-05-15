from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Replace with your YOLOv8 model path
model_path = Path("yolov8n.pt")

# Define polygon coordinates
polygon = np.array([
  [0, 1250],
  [1000, 1080],
  [1250, 1080],
  [1500, 0]
])


def is_inside_polygon(point, polygon):
  """
  Checks if a point is inside a given polygon.

  Args:
      point: A list containing the x and y coordinates of the point.
      polygon: A numpy array representing the polygon vertices.

  Returns:
      True if the point is inside the polygon, False otherwise.
  """
  # https://stackoverflow.com/a/17306349
  cx, cy = point[0], point[1]
  counter = 0
  for i in range(len(polygon)):
    x1, y1 = polygon[i, 0], polygon[i, 1]
    x2, y2 = polygon[(i + 1) % len(polygon), 0], polygon[(i + 1) % len(polygon), 1]
    # Check if line segment intersects with ray from point to positive infinity
    if (y1 < cy <= y2) or (y2 < cy <= y1):
      if x1 + (cy - y1) / (y2 - y1) * (x2 - x1) > cx:
        counter += 1
  return counter % 2 == 1


def detect_in_polygon(image):
  """
  Detects objects within the defined polygon using YOLOv8.

  Args:
      image: The image to process.

  Returns:
      A list of detections within the polygon. Each detection is a dictionary
      containing "name", "conf", "xyxy" (bounding box coordinates).
  """
  # Load YOLOv8 model
    model = YOLO("yolov8n.pt")

  # Perform object detection
    classes, confidences, boxes = model.detect(image)

  # Filter detections within polygon
  polygon_detections = []
  for (classId, conf, x, y, w, h) in zip(classes, confidences, boxes):
    # Extract bounding box center
    center_x, center_y = (int(x - w / 2), int(y - h / 2))

    # Check if center is inside polygon
    if is_inside_polygon([center_x, center_y], polygon):
      # Extract class name from labels file (assuming it exists)
      class_name = model.getLabelNames()[classId]

      # Add detection to polygon detections
      polygon_detections.append({
          "name": class_name,
          "conf": float(conf),
          "xyxy": [int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)]
      })

  return polygon_detections


# Load your image
image = cv2.imread("people.png")

# Detect objects within polygon
polygon_objects = detect_in_polygon(image.copy())

# Draw bounding boxes and labels (optional) - modify for polygon drawing
for obj in polygon_objects:
  x_min, y_min, x_max, y_max = obj["xyxy"]
  cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
  cv2.putText(image, f"{obj['name']} {obj['conf']:.2f}", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX,
              0.7, (0, 0, 255), 2)

# Display the image with detections (optional)
cv2.imshow("Image with Polygon Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
