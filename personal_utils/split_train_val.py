import os
import shutil
import random
import argparse

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def copy_files(file_list, src_image_dir, src_label_dir, dst_image_dir, dst_label_dir):
    for image_file, label_file in file_list:
        shutil.copy(os.path.join(src_image_dir, image_file), os.path.join(dst_image_dir, image_file))
        shutil.copy(os.path.join(src_label_dir, label_file), os.path.join(dst_label_dir, label_file))

def split_dataset(src_image_root, src_label_root, output_root, train_ratio):
    for subdir in os.listdir(src_image_root):
        image_dir = os.path.join(src_image_root, subdir)
        label_dir = os.path.join(src_label_root, subdir)
        
        if os.path.isdir(image_dir) and os.path.isdir(label_dir):
            train_image_dir = os.path.join(output_root, 'images', 'train', subdir)
            train_label_dir = os.path.join(output_root, 'labels', 'train', subdir)
            val_image_dir = os.path.join(output_root, 'images', 'val', subdir)
            val_label_dir = os.path.join(output_root, 'labels', 'val', subdir)

            create_dir(train_image_dir)
            create_dir(train_label_dir)
            create_dir(val_image_dir)
            create_dir(val_label_dir)

            image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
            label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

            assert len(image_files) == len(label_files), f"图像文件和标签文件数量不一致: {subdir}"

            file_pairs = list(zip(image_files, label_files))
            random.shuffle(file_pairs)

            num_files = len(file_pairs)
            num_train = int(num_files * train_ratio)

            train_files = file_pairs[:num_train]
            val_files = file_pairs[num_train:]

            copy_files(train_files, image_dir, label_dir, train_image_dir, train_label_dir)
            copy_files(val_files, image_dir, label_dir, val_image_dir, val_label_dir)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="划分数据集为训练集和验证集")
    # parser.add_argument('--src_image_root', type=str, required=True, help="图片文件的根目录")
    # parser.add_argument('--src_label_root', type=str, required=True, help="标签文件的根目录")
    # parser.add_argument('--output_root', type=str, required=True, help="输出目录")
    # parser.add_argument('--train_ratio', type=float, default=0.9, help="训练集比例")
    
    # args = parser.parse_args()
    
    # split_dataset(args.src_image_root, args.src_label_root, args.output_root, args.train_ratio)
    
    src_image_root = 'E:/HDD/BYD/data/bus_data/B12E03数据集/预标注数据'
    src_label_root = 'E:/HDD/BYD/data/bus_data/B12E03数据集/预标注数据'
    output_root = 'E:/HDD/BYD/data/bus_data/B12E03数据集/预标注数据'
    train_ratio = 0.9

    split_dataset(src_image_root, src_label_root, output_root, train_ratio)
    print("数据集划分完成！")
