from ultralytics import YOLO

def main():
    # Load a model
    # model = YOLO('yolov8n-pose.yaml')  # build a new model from YAML
    model = YOLO('runs/pose/预标注/weights/best.pt')  # load a pretrained model (recommended for training)
    #model = YOLO('yolov8n-obb.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # Train the model
    results = model(source='E:/HDD/BYD/data/bus_data/B12E03数据集/垂直车位', show=False, save=True, save_txt=True, save_conf=True, show_labels=False, show_conf=True, show_boxes=False)


if __name__ == '__main__':
    main()
