from ultralytics import YOLOv10
from ultralytics import YOLO

# Load a model
model = YOLOv10('yolov10n.yaml')  # build a new model from YAML
# model = YOLOv10('yolov8n-obb.pt')  # load a pretrained model (recommended for training)
#model = YOLO('yolov8n-obb.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='od-detect.yaml', epochs=100, imgsz=640, batch=2, save_period=10)