import json
import os

bus_parking_spot_class_map = {
    "vertical_parkset": 0,
    "horizontal_parkset": 1
}

def convert_labelme_to_yolov8_pose(labelme_json_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for json_file in os.listdir(labelme_json_dir):
        if json_file.endswith('.json'):
            labelme_path = os.path.join(labelme_json_dir, json_file)
            with open(labelme_path, 'r', encoding='utf-8') as f:
                labelme_data = json.load(f)

            image_height = labelme_data['imageHeight']
            image_width = labelme_data['imageWidth']
            
            yolo_annotations = []

            for shape in labelme_data['shapes']:
                label = shape['label']
                points = shape['points']
                
                # Calculate bounding box
                x_min = min(point[0] for point in points)
                y_min = min(point[1] for point in points)
                x_max = max(point[0] for point in points)
                y_max = max(point[1] for point in points)
                
                bbox_width = max(min((x_max - x_min) / image_width, 1), 0)
                bbox_height = max(min((y_max - y_min) / image_height, 1), 0)
                center_x = max(min((x_min + x_max) / 2 / image_width, 1), 0)
                center_y = max(min((y_min + y_max) / 2 / image_height, 1), 0)
                
                # Normalize keypoints for YOLOv8
                keypoints = []
                for point in points:
                    x = max(min(point[0] / image_width, 1), 0)
                    y = max(min(point[1] / image_height, 1), 0)
                    keypoints.extend([x, y])
                
                yolo_annotation = f"{bus_parking_spot_class_map[label]} {center_x} {center_y} {bbox_width} {bbox_height} " + " ".join(map(str, keypoints))
                
                yolo_annotations.append(yolo_annotation)
            
            output_file = os.path.join(output_dir, os.path.splitext(json_file)[0] + '.txt')
            with open(output_file, 'w') as out_f:
                for annotation in yolo_annotations:
                    out_f.write(f"{annotation}\n")

if __name__ == '__main__':
    labelme_json_dir = 'E:/HDD/BYD/data/bus_data/B12E03数据集/all'
    output_dir = 'E:/HDD/BYD/data/bus_data/B12E03数据集/水平车位'
    convert_labelme_to_yolov8_pose(labelme_json_dir, output_dir)
