import os
import xml.etree.ElementTree as ET

def convert_voc_to_yolo_obb(voc_dir, yolo_dir):
    # 确保输出目录存在
    os.makedirs(yolo_dir, exist_ok=True)

    # 遍历VOC格式的XML文件
    for xml_file in os.listdir(voc_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(voc_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 获取图像的宽高
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            # 生成对应的YOLO格式的txt文件
            yolo_file = os.path.join(yolo_dir, os.path.splitext(xml_file)[0] + '.txt')
            with open(yolo_file, 'w') as f:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    # 将类别名称转换为类别ID，假设你有一个类别名称到ID的映射
                    class_id = name_to_id(class_name)

                    keypoints = obj.find('keypoints')
                    if keypoints is not None:
                        x0 = float(keypoints.find('x0').text)
                        y0 = float(keypoints.find('y0').text)
                        x1 = float(keypoints.find('x1').text)
                        y1 = float(keypoints.find('y1').text)
                        x2 = float(keypoints.find('x2').text)
                        y2 = float(keypoints.find('y2').text)
                        x3 = float(keypoints.find('x3').text)
                        y3 = float(keypoints.find('y3').text)

                        # 将坐标转换为相对坐标并归一化
                        x0 /= width
                        y0 /= height
                        x1 /= width
                        y1 /= height
                        x2 /= width
                        y2 /= height
                        x3 /= width
                        y3 /= height

                        # 写入YOLO格式：class_id x0 y0 x1 y1 x2 y2 x3 y3
                        f.write(f"{class_id} {x0} {y0} {x1} {y1} {x2} {y2} {x3} {y3}\n")

def name_to_id(name):
    # 你需要定义一个类别名称到ID的映射
    # 例如：
    class_dict = {
        "parking_empty": 0,
        "wheel_stopper": 1
        # 添加更多类别
    }
    return class_dict.get(name, -1)

if __name__ == "__main__":
    voc_root = "/mnt/liujiaren/data/psd_data/annos"
    yolo_root = "/mnt/liujiaren/data/psd_data/yolo_obb_labels"
    for dataset in os.listdir(voc_root):
        voc_dir = os.path.join(voc_root, dataset) 
        yolo_dir = os.path.join(yolo_root, dataset)
        convert_voc_to_yolo_obb(voc_dir, yolo_dir)
