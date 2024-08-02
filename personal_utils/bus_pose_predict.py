import os
import gc
from ultralytics import YOLO
from personal_utils import convert_yolo_to_labelme

def main():
    # Load a model
    # model = YOLO('yolov8n-pose.yaml')  # build a new model from YAML
    model = YOLO('runs/pose/预标注/weights/best.pt')  # load a pretrained model (recommended for training)
    #model = YOLO('yolov8n-obb.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # # Train the model
    # results = model(source='/home/user/work/data/bus_data/B12E03数据集/垂直车位', show=False, save=True, save_txt=True, save_conf=True, show_labels=False, show_conf=True, show_boxes=False)
    
    preditc_imamge_dir = '/home/user/work/data/bus_data/B12E03数据集/待预标注数据/srv'
    yolo_dir = "runs/pose/待预标注"
    labelme_dir = '/home/user/work/data/bus_data/B12E03数据集/待预标注数据/srv'    
    
    for sub_dir in os.listdir(preditc_imamge_dir):

        if os.path.exists(os.path.join(yolo_dir, sub_dir)):
            continue
        
        cur_image_dir = os.path.join(preditc_imamge_dir, sub_dir)
        if os.path.isdir(cur_image_dir):
            model(source=cur_image_dir, show=False, save=False, save_txt=True, save_conf=True, show_labels=False, show_conf=False, show_boxes=False, project='runs/pose/待预标注', name=sub_dir)
        
        # Clear memory after processing each folder
        del model # Delete the model object
        gc.collect() # Force garbage collection
        
        # Reload model for the next folder
        model = YOLO('runs/pose/预标注/weights/best.pt')
        
        # Convert yolo txt label to labelme json
        yolo_subdir = os.path.join(yolo_dir, sub_dir, 'labels')
        labelme_subdir = os.path.join(labelme_dir, sub_dir)
        
        if os.path.isdir(yolo_subdir):
            os.makedirs(labelme_subdir, exist_ok=True)  
            convert_yolo_to_labelme.convert_main(yolo_subdir, labelme_subdir, 640, 640)        


if __name__ == '__main__':
    main()
