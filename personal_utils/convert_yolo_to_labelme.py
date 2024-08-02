import os
import json

class_mapping = {
  0: "cycle",
  1: "pedestrian",
  2: "speed-bump",
  3: "parking_lock_down",
  4: "parking_lock_up",
  5: "traffic_cone",
  6: "no_parking_sign",
  7: "warning_post",
  8: "carton",
  9: "seat",
  10: "bucket",
  11: "vehicle",
  12: "wheer_bar"      
}

def clip(src, image_weight=1920, image_height=1300):
    dst = [min(max(src[0], 0), image_weight), min(max(src[1], 0), image_height)]
    return dst

def yolo_to_labelme_format(yolo_file, image_width=1920, image_height=1300):
    with open(yolo_file, 'r') as f:
        yolo_labels = f.readlines()

    shapes = []
    for line in yolo_labels:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1]) * image_width
        y_center = float(parts[2]) * image_height
        width = float(parts[3]) * image_width
        height = float(parts[4]) * image_height

        # # 强制将坐标值限制在0到1之间
        # x_center = min(max(x_center, 0), 1)
        # y_center = min(max(y_center, 0), 1)
        # width = min(max(width, 0), 1)
        # height = min(max(height, 0), 1)
        
        xmin = x_center - width / 2
        ymin = y_center - height / 2
        xmax = x_center + width / 2
        ymax = y_center + height / 2


        shape = {
            "label": class_mapping[class_id],
            "points": [
                [xmin, ymin],
                [xmax, ymax]
            ],
            "group_id": None,
            "description": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        shapes.append(shape)
    return shapes

def calculate_iou(box1, box2):
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    
    # Calculate the (x, y)-coordinates of the intersection rectangle
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)
    
    # Compute the area of intersection rectangle
    inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)
    
    # Compute the area of both the prediction and ground-truth rectangles
    box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)
    
    # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def is_already_annotated(existing_shapes, new_shape, iou_threshold=0.5):
    new_label = new_shape['label']
    new_box = new_shape['points']
    new_box = [new_box[0][0], new_box[0][1], new_box[1][0], new_box[1][1]]
    
    for shape in existing_shapes:
        if shape['label'] == new_label:
            existing_box = shape['points']
            existing_box = [existing_box[0][0], existing_box[0][1], existing_box[1][0], existing_box[1][1]]
            if calculate_iou(existing_box, new_box) >= iou_threshold:
                return True
    return False

def merge_yolo_into_labelme(yolo_dir, labelme_dir):
    for root, dirs, files in os.walk(yolo_dir):
        for file in files:
            if file.endswith('.txt'):
                txt_file = os.path.join(root, file)
                
                # 计算相对于yolo_dir的相对路径
                relative_path = os.path.relpath(txt_file, yolo_dir)
                # 用相对路径来定位对应的JSON文件
                json_file = os.path.join(labelme_dir, os.path.splitext(relative_path)[0] + '.json')
                
                yolo_objects = yolo_to_labelme_format(txt_file)
                
                if not os.path.exists(json_file):
                    labelme_data = {
                        "version": "5.2.1",
                        "flags": {},
                        "shapes": [],
                        "imagePath": file.replace('.txt', '.jpg'),
                        "imageData": None,
                        "imageHeight": 1300,
                        "imageWidth": 1920
                    }                                      
                    print(f"LabelMe JSON file {json_file} does not exist, create.")
                    # continue
                else:
                    with open(json_file, 'r') as f:
                        labelme_data = json.load(f)
                
                for yolo_object in yolo_objects:
                    if not is_already_annotated(labelme_data['shapes'], yolo_object):
                        labelme_data['shapes'].append(yolo_object)

                with open(json_file, 'w') as f:
                    json.dump(labelme_data, f, indent=4)
                print(f"Updated {json_file}")


def convert_yolo_to_labelme(yolo_file, labelme_file, image_width, image_height, image_path):
    with open(yolo_file, 'r') as f:
        yolo_labels = f.readlines()

    shapes = []
    for line in yolo_labels:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1]) * image_width
        y_center = float(parts[2]) * image_height
        width = float(parts[3]) * image_width
        height = float(parts[4]) * image_height

        # 强制将坐标值限制在0到1之间
        x_center = min(max(x_center, 0), image_width)
        y_center = min(max(y_center, 0), image_height)
        width = min(max(width, 0), image_width)
        height = min(max(height, 0), image_height)        

        xmin = x_center - width / 2
        ymin = y_center - height / 2
        xmax = x_center + width / 2
        ymax = y_center + height / 2

        shape = {
            "label": class_mapping[class_id],
            "points": [
                [xmin, ymin],
                [xmax, ymax]
            ],
            "group_id": None,
            "description": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        shapes.append(shape)

    labelme_data = {
        "version": "5.2.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }

    with open(labelme_file, 'w') as f:
        json.dump(labelme_data, f, indent=4)
        
        
def convert_yolo_to_labelme_by_keypoints(yolo_file, labelme_file, image_width, image_height, image_path):
    with open(yolo_file, 'r') as f:
        yolo_labels = f.readlines()

    shapes = []
    for line in yolo_labels:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1]) * image_width
        y_center = float(parts[2]) * image_height
        width = float(parts[3]) * image_width
        height = float(parts[4]) * image_height
        
        kp0 = [float(parts[5]) * image_width, float(parts[6]) * image_height]
        kp1 = [float(parts[7]) * image_width, float(parts[8]) * image_height]
        kp2 = [float(parts[9]) * image_width, float(parts[10]) * image_height]
        kp3 = [float(parts[11]) * image_width, float(parts[12]) * image_height]

        shape = {
            "label": class_mapping[class_id],
            "points": [
                clip(kp0, image_width, image_height),
                clip(kp1, image_width, image_height),
                clip(kp2, image_width, image_height),
                clip(kp3, image_width, image_height)
            ],
            "group_id": None,
            "description": None,
            "shape_type": "polygon",
            "flags": {}
        }
        shapes.append(shape)

    labelme_data = {
        "version": "5.2.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }

    with open(labelme_file, 'w') as f:
        json.dump(labelme_data, f, indent=4)
        

def convert_main(yolo_dir, labelme_dir, image_width=1920, image_height=1300):

    for filename in os.listdir(yolo_dir):
        if filename.endswith('.txt'):
            yolo_file = os.path.join(yolo_dir, filename)
            image_filename = filename.replace('.txt', '.jpg')  # 假设图像文件扩展名为.jpg
            image_path = os.path.join(labelme_dir, image_filename)
            
            # # 假设你知道图像的宽度和高度
            # image_width = 1920
            # image_height = 1300

            labelme_file = os.path.join(labelme_dir, filename.replace('.txt', '.json'))
            # convert_yolo_to_labelme(yolo_file, labelme_file, image_width, image_height, image_filename)
            convert_yolo_to_labelme_by_keypoints(yolo_file, labelme_file, image_width, image_height, image_filename)

    print("转换完成！")        

if __name__ == "__main__":
    yolo_dir = "runs/pose/待预标注"
    labelme_dir = '/home/user/work/data/bus_data/B12E03数据集/已预标注数据_JSON'
    # merge_yolo_into_labelme(yolo_dir, labelme_dir)
    # os.makedirs(labelme_dir, exist_ok=True)
        
    # convert_main(yolo_dir, labelme_dir, 640, 640)

    for subdir in os.listdir(yolo_dir):
        yolo_subdir = os.path.join(yolo_dir, subdir, 'labels')
        labelme_subdir = os.path.join(labelme_dir, subdir)
        
        if os.path.isdir(yolo_subdir):
            os.makedirs(labelme_subdir, exist_ok=True)  
            convert_main(yolo_subdir, labelme_subdir, 640, 640)

