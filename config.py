# config.py - YOLO-World 项目共享常量
"""所有脚本共享的配置、路径和常量。"""

import torch
from pathlib import Path

# ---- 项目根目录 ----
PROJECT_ROOT = Path(__file__).resolve().parent

# ---- 设备 ----
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CUDA_AVAILABLE = torch.cuda.is_available()

# ---- 类别文本（19 个 Cityscapes 类别）----
CLASS_TEXTS = [
    "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light",
    "traffic sign", "building", "vegetation", "sky",
    "road", "ground", "sidewalk", "fence", "bridge", "wall"
]

# ---- 数据集 ----
CITYSCAPES_YAML = "cityscapes.yaml"

# ---- 模型权重路径 ----
WEIGHTS_DIR = PROJECT_ROOT / "weights"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "yolov8s-worldv2.pt"

# ---- 训练 ----
TRAIN_DIR = PROJECT_ROOT / "runs" / "train" / "yolo-world-cityscapes"
TRAIN_WEIGHTS_DIR = TRAIN_DIR / "weights"
BEST_PT = TRAIN_WEIGHTS_DIR / "best.pt"
LAST_PT = TRAIN_WEIGHTS_DIR / "last.pt"

# ---- 基准测试 ----
BENCHMARK_DIR = PROJECT_ROOT / "runs" / "benchmark"

# ---- 验证 ----
VAL_DIR = PROJECT_ROOT / "runs" / "val"

# ---- 预测 ----
PREDICT_DIR = PROJECT_ROOT / "runs" / "predict"

# ---- 数据 ----
DATA_DIR = PROJECT_ROOT / "data"
CITYSCAPES2_DIR = DATA_DIR / "cityscapes2"

# ---- 模型参数估算（用于 compare_models）----
PARAM_ESTIMATES = {
    "yolov8s-world.pt": 11.2,
    "yolov8s-worldv2.pt": 11.2,
    "yolov8m-world.pt": 25.9,
    "yolov8m-worldv2.pt": 25.9,
    "yolov8l-world.pt": 43.7,
    "yolov8l-worldv2.pt": 43.7,
    "yolov8x-world.pt": 68.2,
    "yolov8x-worldv2.pt": 68.2,
}
