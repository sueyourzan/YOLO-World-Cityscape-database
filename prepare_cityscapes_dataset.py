# prepare_cityscapes_dataset.py
"""将 Cityscapes 数据集转换为 YOLO 格式。"""

import json
import shutil
from pathlib import Path
import numpy as np

from config import DATA_DIR, CITYSCAPES2_DIR

# Cityscapes 类别映射
CITYSCAPES_CLASSES = {
    'person': 0, 'rider': 1, 'car': 2, 'truck': 3, 'bus': 4,
    'train': 5, 'motorcycle': 6, 'bicycle': 7, 'traffic light': 8,
    'traffic sign': 9, 'building': 10, 'vegetation': 11, 'sky': 12,
    'road': 13, 'ground': 14, 'sidewalk': 15, 'fence': 16,
    'bridge': 17, 'wall': 18
}


def prepare_cityscapes_dataset():
    """准备 Cityscapes 数据集为 YOLO 格式。"""
    original_dir = DATA_DIR / "cityscapes"
    yolo_dir = CITYSCAPES2_DIR

    print(f"原始数据集目录: {original_dir}")
    print(f"YOLO格式目录: {yolo_dir}")

    for split in ['train', 'val', 'test']:
        (yolo_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    for split in ['train', 'val', 'test']:
        print(f"\n处理 {split} 集...")
        img_source_dir = original_dir / "leftImg8bit" / split

        if not img_source_dir.exists():
            print(f"跳过 {split}: 目录不存在 {img_source_dir}")
            continue

        city_folders = [f for f in img_source_dir.iterdir() if f.is_dir()]

        for city_path in sorted(city_folders):
            for img_file in city_path.iterdir():
                if not img_file.name.endswith('_leftImg8bit.png'):
                    continue

                dst_img_name = f"{city_path.name}_{img_file.name}"
                dst_img_path = yolo_dir / 'images' / split / dst_img_name
                shutil.copy2(str(img_file), str(dst_img_path))

                if split != 'test':
                    gt_file = img_file.name.replace('_leftImg8bit.png', '_gtFine_polygons.json')
                    gt_path = original_dir / "gtFine" / split / city_path.name / gt_file

                    if gt_path.exists():
                        convert_to_yolo(gt_path, yolo_dir, split, dst_img_name)

                print(f"  ✓ 处理: {img_file.name}")

    print(f"\n✅ 数据集准备完成！")
    print(f"YOLO格式数据集在: {yolo_dir}")
    return yolo_dir


def convert_to_yolo(gt_path: Path, yolo_dir: Path, split: str,
                    img_name: str, class_mapping: dict = None):
    """将 Cityscapes 标注转换为 YOLO 格式。"""
    if class_mapping is None:
        class_mapping = CITYSCAPES_CLASSES

    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_width = data['imgWidth']
    img_height = data['imgHeight']

    label_name = img_name.replace('.png', '.txt')
    label_path = yolo_dir / 'labels' / split / label_name

    with open(label_path, 'w', encoding='utf-8') as f:
        for obj in data['objects']:
            label = obj['label']

            if label not in class_mapping:
                continue

            class_id = class_mapping[label]
            polygon = np.array(obj['polygon'])

            x_min, x_max = polygon[:, 0].min(), polygon[:, 0].max()
            y_min, y_max = polygon[:, 1].min(), polygon[:, 1].max()

            x_center = max(0, min(1, (x_min + x_max) / 2 / img_width))
            y_center = max(0, min(1, (y_min + y_max) / 2 / img_height))
            width = max(0, min(1, (x_max - x_min) / img_width))
            height = max(0, min(1, (y_max - y_min) / img_height))

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def check_prepared_dataset(yolo_dir: Path = None):
    """检查准备好的数据集结构。"""
    if yolo_dir is None:
        yolo_dir = CITYSCAPES2_DIR

    print(f"\n检查数据集结构...")
    for split in ['train', 'val']:
        img_dir = yolo_dir / 'images' / split
        label_dir = yolo_dir / 'labels' / split

        if img_dir.exists():
            images = [f for f in img_dir.iterdir() if f.suffix in ('.jpg', '.png')]
            print(f"  {split} 图像: {len(images)} 个")
        else:
            print(f"  ❌ {split} 图像目录不存在")

        if label_dir.exists():
            labels = list(label_dir.glob('*.txt'))
            print(f"  {split} 标签: {len(labels)} 个")
        else:
            print(f"  ❌ {split} 标签目录不存在")


if __name__ == "__main__":
    print("=" * 70)
    print("Cityscapes 数据集准备工具")
    print("=" * 70)

    yolo_dir = prepare_cityscapes_dataset()
    check_prepared_dataset(yolo_dir)
