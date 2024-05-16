import cv2
import matplotlib.pyplot as plt
import numpy as np
# import supervision as sv
from ultralytics import YOLO


model = YOLO("yolov8n.pt")
image = cv2.imread('people.png')
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run Detection
results = model(image)[0]

# initiate polygon zone
polygon = ([
    [1214, 0],
    [831, 918],
    [1110, 901],
    [1368, 0]
])

poly = [(250, 100), (400, 200), (350, 360),
            (150, 360), (100, 200)]

isClosed = True
color = (2550,0,0)
thickness = 50
poly_image = cv2.polylines(rgb_image, [polygon], True, color, thickness)

plt.imshow(poly_image)

%matplotlib qt


'''
https://gist.github.com/inside-code-yt/7064d1d1553a2ee117e60217cfd1d099
'''
from polygenerator import random_polygon
import matplotlib.pyplot as plt


def is_inside(edges, xp, yp):
    cnt = 0
    for edge in edges:
        (x1, y1), (x2, y2) = edge
        if (yp < y1) != (yp < y2) and xp < x1 + ((yp-y1)/(y2-y1))*(x2-x1):
            cnt += 1
    return cnt%2 == 1

is_inside(polygon, 800,100)

def onclick(event):
    xp, yp = event.xdata, event.ydata
    if is_inside(edges, xp, yp):
        print("inside")
        plt.plot(xp, yp, "go", markersize=5)
    else:
        print("outside")
        plt.plot(xp, yp, "ro", markersize=5)
    plt.gcf().canvas.draw()


polygon = random_polygon(num_points=20)
polygon.append(polygon[0])
edges = list(zip(polygon, polygon[1:] + polygon[:1]))
plt.figure(figsize=(10, 10))
plt.gca().set_aspect("equal")
xs, ys = zip(*polygon)
plt.gcf().canvas.mpl_connect('button_press_event', onclick)
plt.plot(xs, ys, "b-", linewidth=0.8)
plt.show()


