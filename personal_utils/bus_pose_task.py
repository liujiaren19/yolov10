from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO('yolov8n-pose.yaml')  # build a new model from YAML
    # model = YOLOv10('yolov8n-obb.pt')  # load a pretrained model (recommended for training)
    #model = YOLO('yolov8n-obb.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # Train the model
    results = model.train(data='bus-pose.yaml', epochs=200, imgsz=640, batch=16, workers=1, save_period=10, amp=False)


if __name__ == '__main__':
    main()
