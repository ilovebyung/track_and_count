from polygenerator import random_polygon
import matplotlib.pyplot as plt
import cv2
import numpy as np
from ultralytics import YOLO
# pip install PyQt5
%matplotlib qt

model = YOLO("yolov8m.pt")
image = cv2.imread('people.png')
# Convert the image from BGR to RGB
rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run Detection
results = model(image)[0]

def is_inside(edges, xp, yp):
    cnt = 0
    for edge in edges:
        (x1, y1), (x2, y2) = edge
        if (yp < y1) != (yp < y2) and xp < x1 + ((yp-y1)/(y2-y1))*(x2-x1):
            cnt += 1
    return cnt%2 == 1

polygon = [((0, 1250),(1000, 1080)),((1250, 1080),(1500, 0))]

is_inside(polygon, 0,2000)

# Draw the circle on the image
center = (100,200)
color = (0, 0, 255)
image = cv2.circle(rgb_img, center, 10, color, 2)

plt.imshow(image, cmap='gray')

# Draw the polygon (closed shape)
# Define the polygon points
pts = np.array([[[0, 1250], [1000, 1080], [1250, 1080], [1500, 0]]], dtype=np.int32)
img = cv2.fillPoly(image, pts, color, 2)