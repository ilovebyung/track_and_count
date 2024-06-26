from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Export the model to ONNX format
#model.export(format="onnx")  # creates 'yolov8n.onnx'

''' CLI 
# Export a YOLOv8n PyTorch model to ONNX format
yolo export model=yolov8n.pt format=onnx  # creates 'yolov8n.onnx'

# Run inference with the exported model
yolo predict model=yolov8n.onnx source='https://ultralytics.com/images/bus.jpg'
'''
# Load the exported ONNX model
onnx_model = YOLO("yolov8n.onnx")

# Run inference
results = onnx_model("https://ultralytics.com/images/bus.jpg")