# YOLO-World Cityscapes 训练与评估工具

基于 [Ultralytics YOLO-World](https://docs.ultralytics.com/models/yolo-world/) 的 Cityscapes 语义分割数据集目标检测全流程工具链。

## 项目结构

```
YOLO-World/
├── config.py                     # 共享常量（路径、类别、设备配置）
├── utils.py                      # 共享工具函数（模型加载、路径查找等）
├── plot_utils.py                 # Matplotlib 绘图配置与通用组件
│
├── prepare_cityscapes_dataset.py # 数据集准备：Cityscapes → YOLO 格式
├── train_yolo_world.py           # 模型训练
├── benchmark_cityscapes.py       # 模型性能基准测试（FPS、内存、多模型对比）
├── validate_cityscapes.py        # 模型验证 / 推理 / 导出
├── visualize_results.py          # 训练 & 验证 & 预测结果可视化
│
├── cityscapes.yaml               # Cityscapes 数据集配置
├── weights/                      # 预训练权重目录
├── data/                         # 数据集目录
│   ├── cityscapes/               # 原始 Cityscapes 数据集
│   └── cityscapes2/              # YOLO 格式 Cityscapes
│       ├── images/  train/ val/ test/
│       └── labels/  train/ val/
├── runs/                         # 输出目录（训练 / 验证 / 预测 / benchmark）
└── yolo11n.pt                    # 基础 YOLO 模型
```

## 环境要求

- Python >= 3.8
- PyTorch >= 1.10（推荐 CUDA 支持）
- ultralytics >= 8.2.0
- matplotlib, pandas, numpy, pyyaml

```bash
pip install ultralytics torch matplotlib pandas numpy pyyaml
```

## 克隆后首次设置

> 仓库不包含数据集和预训练权重（已在 `.gitignore` 中排除）。克隆后需手动准备以下内容：

### 1. 下载 Cityscapes 数据集

从 [Cityscapes 官网](https://www.cityscapes-dataset.com/downloads/) 下载以下文件（需注册账号）：

| 文件 | 内容 | 大小 |
|------|------|------|
| `leftImg8bit_trainvaltest.zip` | 全部图像 (train/val/test) | ~11GB |
| `gtFine_trainvaltest.zip` | 精细标注 (train/val) | ~241MB |

下载后按以下结构放置：

```
data/
└── cityscapes/
    ├── leftImg8bit/
    │   ├── train/
    │   │   ├── aachen/
    │   │   ├── bochum/
    │   │   └── ...
    │   ├── val/
    │   └── test/
    └── gtFine/
        ├── train/
        └── val/
```

```bash
# 解压命令示例
unzip leftImg8bit_trainvaltest.zip -d data/cityscapes/
unzip gtFine_trainvaltest.zip -d data/cityscapes/
```

### 2. 下载预训练权重

从 [Ultralytics 发布页](https://github.com/ultralytics/assets/releases) 下载 YOLO-World 权重，或由 ultralytics 首次运行时自动下载。

手动下载后放置于 `weights/` 目录：

```bash
mkdir -p weights
# 最小可用权重（推荐）
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-worldv2.pt -P weights/
```

### 3. 转换为 YOLO 格式

```bash
python prepare_cityscapes_dataset.py
```

处理完成后 `data/cityscapes2/` 目录即为 YOLO 格式数据集。

---

## 快速开始

### 1. 准备数据集

> 首次使用请先完成上方「克隆后首次设置」。

```bash
python prepare_cityscapes_dataset.py
```

将原始 Cityscapes 数据集（`data/cityscapes/`）转换为 YOLO 格式（`data/cityscapes2/`），包含 19 个类别。

### 2. 训练模型

```bash
python train_yolo_world.py
```

默认使用 `yolov8s-worldv2.pt` 预训练权重，在 Cityscapes 上微调 5 个 epoch。

训练产物保存在 `runs/train/yolo-world-cityscapes/`：
- `weights/best.pt` — 最佳模型权重
- `weights/last.pt` — 最后一轮权重
- `results.csv` — 训练指标记录

### 3. 验证模型

```bash
python validate_cityscapes.py
```

交互式菜单：
- **选项 1**：计算 mAP50 / mAP75 / mAP50-95 / Precision / Recall
- **选项 2**：单张图像推理演示
- **选项 3**：批量推理
- **选项 4**：导出 ONNX / TorchScript / OpenVINO 格式
- **选项 5**：全部执行

### 4. 基准测试

```bash
python benchmark_cityscapes.py
```

交互式菜单：
- **选项 1**：单个模型推理速度测试（320-1024px, FPS, GPU 内存）
- **选项 2**：多模型性能对比（YOLOv8s/m/l/x World v1/v2）
- **选项 3**：可视化已有结果
- **选项 4**：全部执行

输出 HTML 报告 + PNG 图表，保存在 `runs/benchmark/`。

### 5. 结果可视化

```bash
python visualize_results.py
```

交互式菜单：
- **选项 1**：训练曲线可视化（mAP、Precision、Recall、Fitness）
- **选项 2**：验证结果分析（各类别 AP 排行、mAP 对比、雷达图）
- **选项 3**：预测结果分析（类别分布、置信度分布）
- **选项 4**：生成完整分析报告

## 数据集类别（19 类）

| ID | 类别 (EN) | 类别 (CN) |
|----|-----------|-----------|
| 0 | person | 行人 |
| 1 | rider | 骑手 |
| 2 | car | 汽车 |
| 3 | truck | 卡车 |
| 4 | bus | 公交车 |
| 5 | train | 火车 |
| 6 | motorcycle | 摩托车 |
| 7 | bicycle | 自行车 |
| 8 | traffic light | 交通灯 |
| 9 | traffic sign | 交通标志 |
| 10 | building | 建筑 |
| 11 | vegetation | 植被 |
| 12 | sky | 天空 |
| 13 | road | 道路 |
| 14 | ground | 地面 |
| 15 | sidewalk | 人行道 |
| 16 | fence | 围栏 |
| 17 | bridge | 桥梁 |
| 18 | wall | 墙壁 |

## 可用的 YOLO-World 预训练权重

| 模型 | 参数量 | 用途 |
|------|--------|------|
| yolov8s-world.pt | 11.2M | 轻量推理 |
| yolov8s-worldv2.pt | 11.2M | 轻量推理 v2 |
| yolov8m-world.pt | 25.9M | 平衡速度与精度 |
| yolov8m-worldv2.pt | 25.9M | 平衡速度与精度 v2 |
| yolov8l-world.pt | 43.7M | 高精度 |
| yolov8l-worldv2.pt | 43.7M | 高精度 v2 |
| yolov8x-world.pt | 68.2M | 最高精度 |
| yolov8x-worldv2.pt | 68.2M | 最高精度 v2 |

权重文件放置于 `weights/` 目录下，或由 ultralytics 自动下载。

## 训练参数配置

关键训练参数（在 `train_yolo_world.py` 中修改）：

```python
train_args = {
    'epochs': 5,        # 训练轮次
    'imgsz': 640,       # 输入图像尺寸
    'batch': 4,         # 批量大小
    'lr0': 0.0001,      # 初始学习率
    'box': 7.5,         # 框损失权重
    'cls': 0.5,         # 分类损失权重
    'dfl': 1.5,         # DFL 损失权重
    'amp': True,        # 混合精度训练
    'patience': 100,    # 早停耐心值
}
```

## 常见问题

### Windows 系统 `workers` 必须设为 0

DataLoader 在多进程模式下与 Windows 不兼容，项目中已默认配置 `workers=0`。

### OpenVINO 导出问题

OpenVINO 2025+ 版本缺少 `__version__` 属性，`validate_cityscapes.py` 已内置自动修复。如仍有问题，可降级：

```bash
pip install openvino==2024.3.0 openvino-dev==2024.3.0
```

### 中文显示问题

`plot_utils.py` 已配置中文字体回退 (`SimHei → Arial Unicode MS → DejaVu Sans`)。如仍无法显示中文，请安装中文字体：

```bash
# Windows
pip install fonts-simhei  # 或从系统字体目录安装

# Linux
sudo apt install fonts-wqy-microhei
```

## 代码结构说明

项目采用模块化设计，三个共享模块支撑五个业务脚本：

| 模块 | 职责 |
|------|------|
| `config.py` | 所有路径、类别列表、设备配置、模型参数估算 |
| `utils.py` | 工作区初始化、模型加载（含降级逻辑）、路径查找、安全计算 |
| `plot_utils.py` | 全局字体设置、柱状图标签、总结文本框、图表保存、雷达图 |

各业务脚本通过导入共享模块消除代码重复，可在 `runs/` 目录下查看各自输出。
