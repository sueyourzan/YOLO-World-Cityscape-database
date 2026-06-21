# utils.py - YOLO-World 项目共享工具函数
"""跨脚本复用的工具函数。"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional

# 确保项目根目录在 Python 路径中
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def setup_workspace():
    """将工作目录切换到项目根目录，并配置 CUDA。"""
    os.chdir(str(_project_root))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True


def get_device_name() -> str:
    """获取设备名称字符串。"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def load_yolo_world(model_path: str, class_texts: list = None):
    """
    加载 YOLOWorld 模型并可选地设置类别。

    自动处理 YOLOWorld 不可用时的降级。
    """
    try:
        from ultralytics import YOLOWorld
        model = YOLOWorld(model_path)
    except ImportError:
        from ultralytics import YOLO

        class YOLOWorldWrapper:
            def __init__(self, model_path):
                self.model = YOLO(model_path)

            def __getattr__(self, name):
                # 代理到内部 model
                return getattr(self.model, name)

            def set_classes(self, class_texts):
                if hasattr(self.model, 'set_classes'):
                    self.model.set_classes(class_texts)
                return self

            def train(self, **kwargs):
                return self.model.train(**kwargs)

        model = YOLOWorldWrapper(model_path)

    if class_texts:
        model.set_classes(class_texts)

    return model


def find_model_path(*candidates) -> Optional[str]:
    """
    按优先级查找第一个存在的模型文件路径。

    用法:
        path = find_model_path("./runs/.../best.pt", "./runs/.../last.pt", "./weights/yolov8s-worldv2.pt")
    """
    for path in candidates:
        if os.path.exists(str(path)):
            return str(path)
    return None


def find_latest_experiment_dir(base_path: str, prefix_pattern: str = r"(train|val|exp)\d*$") -> Optional[str]:
    """
    智能查找最新的实验目录（按自然顺序排序）。

    Args:
        base_path: 基础搜索路径
        prefix_pattern: 目录名匹配的正则模式

    Returns:
        最新实验目录的完整路径，若未找到则返回 None
    """
    import re

    if not os.path.exists(base_path):
        return None

    dirs = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and re.match(prefix_pattern, item):
            dirs.append(item)

    if not dirs:
        return base_path

    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', s)]

    dirs.sort(key=natural_sort_key)
    return os.path.join(base_path, dirs[-1])


def safe_mean(values) -> float:
    """安全计算均值（处理列表或 numpy 数组）。"""
    if values is None:
        return 0.0
    if isinstance(values, (list, np.ndarray)):
        if len(values) == 0:
            return 0.0
        return float(np.mean(values))
    return float(values)
