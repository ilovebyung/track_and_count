import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from ultralytics import YOLO


model = YOLO("yolov8n.pt")
image = cv2.imread('people.png')

# Run Detection
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)
detections = detections[detections.class_id == 0]
detections = detections[detections.confidence > 0.8]

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(detections['class_name'], detections.confidence)
]

frames_generator = sv.get_video_frames_generator('people.mp4')

for frame in frames_generator:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)

    # annotated_image = label_annotator.annotate(
    #     scene=annotated_image, detections=detections, labels=labels)

    cv2.imshow('annotated_image', annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.destroyAllWindows()    

# initiate polygon zone
polygon = np.array([
    [0, 1250],
    [1000, 1080],
    [1250, 1080],
    [1500, 0]
])


video_info = sv.VideoInfo.from_video_path('people.mp4')
zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)


# initiate annotators
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.white(), thickness=6, text_thickness=6, text_scale=4)

# extract video frame
generator = sv.get_video_frames_generator('people.mp4')
iterator = iter(generator)
frame = next(iterator)

plt.imshow(frame)

# detect
results = model(frame, imgsz=1280)[0]
detections = sv.Detections.from_ultralytics(results)
detections = detections[detections.class_id == 0]
detections = detections[detections.confidence > 0.8]
zone.trigger(detections=detections)

# annotate
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(detections['class_name'], detections.confidence)
]

detections = sv.Detections.from_ultralytics(results)
annotated_image = bounding_box_annotator.annotate(
    scene=frame, detections=detections)

plt.imshow(annotated_image)

# mixed conditions

# initiate polygon zone
polygon = np.array([
    [1214, 0],
    [831, 918],
    [1110, 901],
    [1368, 0]
])

isClosed = True
color = (0,0,255)
thickness = 50
img = cv2.polylines(frame, [polygon], True, color, thickness)
plt.imshow(img)

video_info = sv.VideoInfo.from_video_path('people.mp4')
zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)

results = model(frame)[0]
# detections = sv.Detections.from_ultralytics(results)
# detections = detections[detections.class_id == 0]
# detections = detections[detections.confidence > 0.8]
mask = zone.trigger(detections=detections)
detections = detections[(detections.confidence > 0.8) & mask] 
annotated_image = bounding_box_annotator.annotate(
    scene=frame, detections=detections)

plt.imshow(annotated_image)




# Create a new figure window
plt.figure()

# Display the image
plt.imshow(image)

# Show the image in a separate window
plt.show()

%matplotlib inline
%matplotlib qt
plt.im