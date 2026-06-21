# train_yolo_world.py
"""YOLO-World Cityscapes 训练脚本。"""

import os
import traceback
import pandas as pd
import yaml

from config import (
    CLASS_TEXTS, CITYSCAPES_YAML, DEVICE, CUDA_AVAILABLE,
    DEFAULT_WEIGHTS, TRAIN_DIR, TRAIN_WEIGHTS_DIR, LAST_PT,
)
from utils import setup_workspace, load_yolo_world


def train_yolo_world():
    """执行 YOLO-World 模型训练。"""
    try:
        weights_path = str(DEFAULT_WEIGHTS) if DEFAULT_WEIGHTS.exists() else "yolov8s-worldv2.pt"
        print(f"加载模型: {weights_path}")
        model = load_yolo_world(weights_path, CLASS_TEXTS)
        print(f"设置 {len(CLASS_TEXTS)} 个类别")

        train_args = {
            'data': CITYSCAPES_YAML,
            'epochs': 5,
            'imgsz': 640,
            'batch': 4,
            'amp': True,
            'workers': 0,
            'device': DEVICE,
            'save': True,
            'project': str(TRAIN_DIR.parent),
            'name': TRAIN_DIR.name,
            'exist_ok': True,
            'pretrained': True,
            'lr0': 0.0001,
            'val': True,
            'verbose': True,
            'save_period': 1,
            'save_json': True,
            'save_hybrid': True,
            'plots': True,
            'label_smoothing': 0.0,
            'patience': 100,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
        }

        print("开始训练...")
        print(f"计划训练轮次: {train_args['epochs']}")
        results = model.train(**train_args)
        print("训练完成！")

        check_results_file()
        return results

    except Exception as e:
        print(f"训练出错: {e}")
        traceback.print_exc()
        return None


def check_results_file():
    """检查训练结果文件并确保 epoch 信息完整。"""
    try:
        results_csv = TRAIN_DIR / "results.csv"
        if not results_csv.exists():
            return

        print(f"\n检查结果文件: {results_csv}")
        df = pd.read_csv(str(results_csv))
        print(f"文件包含 {len(df)} 行数据")

        if 'epoch' not in df.columns:
            print("结果文件缺少epoch列，正在修复...")
            args_file = TRAIN_DIR / "args.yaml"
            total_epochs = 5
            if args_file.exists():
                with open(args_file, 'r') as f:
                    args = yaml.safe_load(f)
                total_epochs = args.get('epochs', 5)

            df['epoch'] = range(1, len(df) + 1)
            df.to_csv(str(results_csv), index=False)
            print(f"已添加epoch列，共{len(df)}个epoch")

            if len(df) < total_epochs:
                print(f"警告: 计划训练{total_epochs}个epoch，但只有{len(df)}个epoch的数据")
                print("可能原因:\n1. 训练被中断\n2. 早停机制触发\n3. 保存配置问题")
        else:
            print(f"epoch列已存在，共{df['epoch'].max()}个epoch")

        print("\n训练结果统计:")
        for column in df.columns:
            if column != 'epoch' and pd.api.types.is_numeric_dtype(df[column]):
                if len(df) > 1:
                    print(f"{column:20s}: 均值={df[column].mean():.4f}, "
                          f"范围=[{df[column].min():.4f}, {df[column].max():.4f}]")
                else:
                    print(f"{column:20s}: 值={df[column].iloc[0]:.4f}")

    except Exception as e:
        print(f"检查结果文件时出错: {e}")


def ensure_complete_training():
    """如果存在中断的训练，尝试从中断点继续。"""
    try:
        if not TRAIN_WEIGHTS_DIR.exists() or not LAST_PT.exists():
            return None

        print("发现未完成的训练，尝试继续训练...")
        args_file = TRAIN_DIR / "args.yaml"
        if not args_file.exists():
            return None

        with open(args_file, 'r') as f:
            args = yaml.safe_load(f)

        results_csv = TRAIN_DIR / "results.csv"
        completed_epochs = 0
        if results_csv.exists():
            df = pd.read_csv(str(results_csv))
            if 'epoch' in df.columns:
                completed_epochs = int(df['epoch'].max())

        total_epochs = args.get('epochs', 5)
        remaining_epochs = total_epochs - completed_epochs

        if remaining_epochs <= 0:
            return None

        print(f"已完成 {completed_epochs}/{total_epochs} 个epoch，继续训练 {remaining_epochs} 个epoch")

        model = load_yolo_world(str(LAST_PT))
        continue_args = {
            'resume': True,
            'epochs': total_epochs,
            'device': DEVICE,
        }
        return model.train(**continue_args)

    except Exception as e:
        print(f"继续训练失败: {e}")
        return None


def main():
    print("=" * 70)
    print("YOLO-World Cityscapes 训练")
    print("=" * 70)

    setup_workspace()

    yaml_file = CITYSCAPES_YAML
    if os.path.exists(yaml_file):
        print(f"找到配置文件: {yaml_file}")
    else:
        print(f"缺少配置文件: {yaml_file}")
        return

    continued_results = ensure_complete_training()
    if continued_results is not None:
        print("继续训练完成！")
        return

    train_yolo_world()


if __name__ == "__main__":
    main()
