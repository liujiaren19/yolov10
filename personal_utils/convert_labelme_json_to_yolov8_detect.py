import os
import json

def convert_labelme_to_yolo(labelme_file, yolo_file):
    with open(labelme_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    image_width = data['imageWidth']
    image_height = data['imageHeight']
    
    yolo_labels = []
    
    for shape in data['shapes']:
        label = shape['label']
        if label not in class_mapping:
            continue
        
        class_id = class_mapping[label]
        points = shape['points']
        xmin = min(p[0] for p in points)
        ymin = min(p[1] for p in points)
        xmax = max(p[0] for p in points)
        ymax = max(p[1] for p in points)
        
        x_center = (xmin + xmax) / 2.0 / image_width
        y_center = (ymin + ymax) / 2.0 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height
        
        yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
    with open(yolo_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(yolo_labels))
    
def main():

    # LabelMe JSON 文件所在目录
    labelme_dir = "/path/to/labelme/json/files"
    # 输出的 YOLO 标签文件所在目录
    yolo_dir = "/path/to/yolo/label/files"
    # 类别映射字典，key为类别名称，value为类别ID
    class_mapping = {
        "pedestrian": 0,
        "speed-bump": 1,
        "traffic_cone": 2,
        "no_parking_sign": 3,
        "warning_post": 4,
        "carton": 5,
        "bucket": 6,
        "vehicle": 7,
        "wheer_bar": 8                
    }

    if not os.path.exists(yolo_dir):
        os.makedirs(yolo_dir)

    for filename in os.listdir(labelme_dir):
        if filename.endswith('.json'):
            labelme_file = os.path.join(labelme_dir, filename)
            yolo_file = os.path.join(yolo_dir, os.path.splitext(filename)[0] + '.txt')
            convert_labelme_to_yolo(labelme_file, yolo_file)
    print("转换完成！")

if __name__ == "__main__":
    main()
