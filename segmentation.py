import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from ultralytics import YOLO


model = YOLO("yolov8n-seg.pt")
image = cv2.imread('people.png')

# Run Detection
results = model(image)[0]

result = results[0].plot()
plt.imshow(result)

%matplotlib qt

# Extract Masks
new_result  = results[0]    # first detection    
new_result 
new_result.masks.xyn

extracted_masks = new_result.masks.data
extracted_masks.shape
masks_array = extracted_masks.cpu().numpy()
plt.imshow(masks_array[0], cmap='gray')
