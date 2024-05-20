import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

image = cv2.imread('people.png')
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# initiate polygon zone with a numPy array of points
polygon = np.array([
    [1100, 10],
    [1368, 10],
    [1110, 910],
    [820, 910]
], dtype=np.int32)  # Ensure data type is int32

isClosed = True
color = (250,0,0)
thickness = 10
poly_image = cv2.polylines(rgb_image, [polygon], True, color, thickness)

plt.imshow(poly_image)

# %matplotlib qt

polygon = np.array([
    ([800, 910],    [1100, 10]),
    ([1010, 910],   [1368, 10])
], dtype=np.int32)  # Ensure data type is int32

def is_inside(edges, xp, yp):
    cnt = 0
    for edge in edges:
        (x1, y1), (x2, y2) = edge
        if (yp < y1) != (yp < y2) and xp < x1 + ((yp-y1)/(y2-y1))*(x2-x1):
            cnt += 1
    return cnt%2 == 1

# detect a point inside
is_inside(polygon, 1000,800)
point = (1000,800)
cv2.circle(rgb_image,point,2,color,thickness)
plt.imshow(rgb_image)

# detect a point outside
is_inside(polygon, 400,800)
point = (400,800)
cv2.circle(rgb_image,point,2,color,thickness)
plt.imshow(rgb_image)


# Run Detection in image
model = YOLO('yolov8m.pt')
result = model(image)[0]

for box in result.boxes:
    class_id = result.names[box.cls[0].item()]
    cords = box.xyxy[0].tolist()
    cords = [round(x) for x in cords]
    conf = round(box.conf[0].item(), 2)

    print("Object type:", class_id)
    print("Coordinates:", cords)
    print("Probability:", conf)
    print("---")

    start = cords[0:2]  # x1,y1
    end = cords[2:4]  # x2,y2
    image = cv2.rectangle(image, start, end, color, 2)

rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_img)


# Run Detection in roi
model = YOLO('yolov8s.pt')
result = model(image)[0]

for box in result.boxes:
    class_id = result.names[box.cls[0].item()]
    cords = box.xyxy[0].tolist()
    x1,y1,x2,y2 = [round(x) for x in cords]
    print(class_id, x1,y1)

    if (class_id == 'person') and is_inside(polygon, x1,y1):

      cords = [round(x) for x in cords]
      conf = round(box.conf[0].item(), 2)

      print("Object type:", class_id)
      print("Coordinates:", cords)
      print("Probability:", conf)
      print("---")

      start = cords[0:2]  # x1,y1
      end = cords[2:4]  # x2,y2
      image = cv2.rectangle(image, start, end, color, 8)

rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_img)
