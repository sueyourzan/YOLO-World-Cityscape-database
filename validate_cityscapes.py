# validate_cityscapes.py
"""YOLO-World Cityscapes 模型验证、推理与导出。"""

import os
import json
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    CLASS_TEXTS, CITYSCAPES_YAML, DEVICE, CUDA_AVAILABLE,
    BEST_PT, LAST_PT, DEFAULT_WEIGHTS, VAL_DIR, PREDICT_DIR,
)
from utils import setup_workspace, get_device_name, find_model_path, load_yolo_world, safe_mean
from plot_utils import setup_plot_style, add_bar_labels, create_summary_textbox, save_and_close


setup_plot_style()


# ---- 模型加载辅助 ----

def _load_model_for_validation():
    """统一的模型加载逻辑：查找、加载、设置类别。"""
    model_path = find_model_path(str(BEST_PT), str(LAST_PT), str(DEFAULT_WEIGHTS))
    if model_path is None:
        print("未找到模型文件")
        return None, None
    print(f"找到模型文件: {model_path}")
    print(f"\n加载模型: {model_path}")
    try:
        model = load_yolo_world(model_path, CLASS_TEXTS)
        print(f"已设置 {len(CLASS_TEXTS)} 个类别")
        return model, model_path
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None, None


# ---- 验证 ----

def validate_model():
    """验证训练好的 YOLO-World 模型。"""
    print("=" * 70)
    print("YOLO-World Cityscapes 模型验证")
    print("=" * 70)

    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {CUDA_AVAILABLE}")
        if CUDA_AVAILABLE:
            print(f"GPU: {get_device_name()}")
    except Exception:
        print("PyTorch: 未安装")
        return False

    model, model_path = _load_model_for_validation()
    if model is None:
        return False

    if not os.path.exists(CITYSCAPES_YAML):
        print(f"配置文件不存在: {CITYSCAPES_YAML}")
        return False

    print(f"\n使用配置文件: {CITYSCAPES_YAML}")

    os.makedirs(str(VAL_DIR), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    val_run_dir = VAL_DIR / f"val_{timestamp}"
    val_run_dir.mkdir(parents=True, exist_ok=True)
    print(f"验证结果将保存到: {val_run_dir}")

    val_args = {
        'data': CITYSCAPES_YAML,
        'imgsz': 640,
        'batch': 4,
        'device': DEVICE,
        'workers': 0,
        'conf': 0.001,
        'iou': 0.6,
        'split': 'val',
        'save_json': True,
        'save_hybrid': True,
        'max_det': 300,
        'verbose': True,
        'save': True,
        'plots': True,
        'project': str(VAL_DIR),
        'name': f"val_{timestamp}",
        'exist_ok': True,
    }

    print("\n开始验证...")
    try:
        metrics = model.val(**val_args)
        print(f"\n调试信息 - metrics类型: {type(metrics)}")

        results_data = _extract_metrics(metrics, val_run_dir)
        print_validation_results(results_data)

        save_results(model_path, val_run_dir, results_data)
        generate_visualizations(results_data, val_run_dir)
        return metrics

    except Exception as e:
        print(f"验证出错: {e}")
        traceback.print_exc()
        return None


def _extract_metrics(metrics, val_run_dir) -> dict:
    """从 metrics 对象和文件系统提取验证指标。"""
    results_data = {}

    if not hasattr(metrics, 'box'):
        return results_data

    box = metrics.box
    results_data['box'] = {
        'map': float(box.map) if hasattr(box, 'map') else None,
        'map50': float(box.map50) if hasattr(box, 'map50') else None,
        'map75': float(box.map75) if hasattr(box, 'map75') else None,
    }

    # Precision & Recall — 多种方式尝试获取
    precision_val = getattr(box, 'precision', None)
    recall_val = getattr(box, 'recall', None)

    if hasattr(metrics, 'results') and (precision_val is None or recall_val is None):
        rd = metrics.results
        if precision_val is None:
            precision_val = rd.get('precision')
        if recall_val is None:
            recall_val = rd.get('recall')

    results_data['box']['precision'] = safe_mean(precision_val) if precision_val is not None else None
    results_data['box']['recall'] = safe_mean(recall_val) if recall_val is not None else None

    # Per-class AP
    if hasattr(box, 'maps'):
        class_maps = box.maps
        if isinstance(class_maps, (np.ndarray, list)):
            results_data['box']['maps'] = [float(x) for x in class_maps]

    # 兜底：从 JSON 文件读取
    json_files = sorted(Path(val_run_dir).glob("*.json"), key=os.path.getctime, reverse=True)
    if json_files:
        try:
            with open(json_files[0], 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            for key in ['precision', 'recall']:
                if key in json_data and results_data['box'].get(key) is None:
                    results_data['box'][key] = safe_mean(json_data[key])
        except Exception as e:
            print(f"从JSON文件读取精确率/召回率失败: {e}")

    return results_data


def print_validation_results(results_data: dict):
    """打印验证结果到控制台。"""
    print("\n" + "=" * 70)
    print("验证结果:")
    print("=" * 70)

    box = results_data.get('box', {})
    if 'map50' in box and box['map50'] is not None:
        print(f"mAP50:    {box['map50']:.4f}")
    if 'map75' in box and box['map75'] is not None:
        print(f"mAP75:    {box['map75']:.4f}")
    if 'map' in box and box['map'] is not None:
        print(f"mAP50-95: {box['map']:.4f}")
    if 'precision' in box and box['precision'] is not None:
        print(f"精确度 (Precision): {box['precision']:.4f}")
    if 'recall' in box and box['recall'] is not None:
        print(f"召回率 (Recall): {box['recall']:.4f}")

    if 'maps' in box and box['maps']:
        print("\n各类别 AP50-95 (前10个):")
        for i, class_map in enumerate(box['maps'][:10]):
            name = CLASS_TEXTS[i] if i < len(CLASS_TEXTS) else f"类别 {i}"
            print(f"  {name}: {class_map:.4f}")


def save_results(model_path, val_run_dir, results_data):
    """保存验证结果到文本和 JSON 文件。"""
    results_file = val_run_dir / "validation_results.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("YOLO-World Cityscapes 验证结果\n")
        f.write("=" * 60 + "\n")
        f.write(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型路径: {model_path}\n\n")

        if 'box' in results_data:
            f.write("检测性能指标:\n")
            f.write("-" * 40 + "\n")
            box = results_data['box']
            for key, label in [('map', 'mAP50-95'), ('map50', 'mAP50'), ('map75', 'mAP75'),
                               ('precision', '精确度'), ('recall', '召回率')]:
                if key in box and box[key] is not None:
                    f.write(f"{label}: {box[key]:.4f}\n")
        f.write("\n" + "=" * 60 + "\n")
    print(f"\n验证结果已保存到: {results_file}")

    json_file = val_run_dir / "validation_metrics.json"
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        print(f"指标已保存到JSON文件: {json_file}")
    except Exception as e:
        print(f"保存JSON文件失败: {e}")


def generate_visualizations(results_data, val_run_dir):
    """生成验证结果的可视化图表。"""
    print("\n 生成可视化图表...")

    viz_dir = val_run_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    box_data = results_data.get('box')
    if not box_data:
        print("  没有检测指标数据，无法生成图表")
        return

    map_metrics, map_labels, map_colors = [], [], ['#2E86AB', '#A23B72', '#F18F01']
    for key, label in [('map50', 'mAP50'), ('map75', 'mAP75'), ('map', 'mAP50-95')]:
        if key in box_data and box_data[key] is not None:
            map_metrics.append(box_data[key])
            map_labels.append(label)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1.1 mAP 指标对比
    ax1 = axes[0, 0]
    if map_metrics:
        bars1 = ax1.bar(range(len(map_metrics)), map_metrics,
                        color=map_colors[:len(map_metrics)])
        ax1.set_xticks(range(len(map_metrics)))
        ax1.set_xticklabels(map_labels, fontsize=12)
        ax1.set_ylabel('分数', fontsize=12)
        ax1.set_title('mAP指标对比', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, max(map_metrics) * 1.2 if map_metrics else 1.0)
        add_bar_labels(ax1, bars1, map_metrics, fmt="{:.3f}", fontsize=11)

    # 1.2 精确率-召回率雷达图
    ax2 = axes[0, 1]
    if box_data.get('precision') is not None and box_data.get('recall') is not None:
        metrics_radar = [box_data['precision'], box_data['recall']]
        radar_labels = ['精确率', '召回率']
        from plot_utils import build_radar_chart
        build_radar_chart(ax2, metrics_radar, radar_labels, '精确率-召回率雷达图')
    else:
        ax2.set_visible(False)

    # 1.3 各类别 AP 条形图
    ax3 = axes[1, 0]
    if box_data.get('maps'):
        class_maps = list(box_data['maps'])
        top_n = min(15, len(class_maps))
        top_indices = np.argsort(class_maps)[-top_n:][::-1]

        top_classes, top_values = [], []
        for idx in top_indices:
            top_values.append(class_maps[idx])
            top_classes.append(CLASS_TEXTS[idx] if idx < len(CLASS_TEXTS) else f'类别{idx}')

        colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(top_values)))
        bars3 = ax3.barh(range(len(top_values)), top_values, color=colors)
        ax3.set_yticks(range(len(top_values)))
        ax3.set_yticklabels(top_classes, fontsize=10)
        ax3.set_xlabel('AP分数', fontsize=12)
        ax3.set_title(f'Top {top_n} 类别AP', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.set_xlim(0, max(top_values) * 1.2 if top_values else 1.0)
        for i, value in enumerate(top_values):
            ax3.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=9)

    # 1.4 性能指标总结
    ax4 = axes[1, 1]
    summary_text = "验证结果总结\n\n"
    for key, label in [('map', 'mAP50-95'), ('map50', 'mAP50'), ('map75', 'mAP75'),
                        ('precision', '精确率'), ('recall', '召回率')]:
        if key in box_data and box_data[key] is not None:
            summary_text += f"• {label}: {box_data[key]:.4f}\n"

    if box_data.get('maps'):
        class_maps = list(box_data['maps'])
        if class_maps:
            best_idx = np.argmax(class_maps)
            worst_idx = np.argmin(class_maps)
            best_name = CLASS_TEXTS[best_idx] if best_idx < len(CLASS_TEXTS) else f'类别{best_idx}'
            worst_name = CLASS_TEXTS[worst_idx] if worst_idx < len(CLASS_TEXTS) else f'类别{worst_idx}'
            summary_text += f"\n• 最佳类别: {best_name} (AP={class_maps[best_idx]:.4f})\n"
            summary_text += f"• 最差类别: {worst_name} (AP={class_maps[worst_idx]:.4f})"

    create_summary_textbox(ax4, summary_text, fontsize=12, facecolor='wheat')
    ax4.set_title('性能指标总结', fontsize=14, fontweight='bold')

    fig.suptitle('YOLO-World Cityscapes 验证结果可视化', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    metrics_plot_path = viz_dir / "validation_metrics_comprehensive.png"
    save_and_close(fig, str(metrics_plot_path), dpi=300)
    print(f" 综合验证指标图已保存: {metrics_plot_path}")

    # 单独保存 mAP 对比图
    if map_metrics:
        fig2, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(map_labels, map_metrics, color=map_colors[:len(map_metrics)])
        ax.set_xlabel('指标', fontsize=12)
        ax.set_ylabel('分数', fontsize=12)
        ax.set_title('YOLO-World Cityscapes mAP指标', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(map_metrics) * 1.2 if map_metrics else 1.0)
        for bar, value in zip(bars, map_metrics):
            ax.text(bar.get_x() + bar.get_width() / 2., value + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        map_plot_path = viz_dir / "mAP_comparison.png"
        save_and_close(fig2, str(map_plot_path), dpi=300)
        print(f" mAP对比图已保存: {map_plot_path}")

    print(f" 所有可视化图表已生成，保存在: {viz_dir}")


# ---- 推理 ----

def _common_predict(source, run_name: str):
    """通用推理流程。"""
    model, _ = _load_model_for_validation()
    if model is None:
        return

    results = model.predict(
        source=source,
        imgsz=640,
        conf=0.25,
        iou=0.5,
        device=DEVICE,
        save=True,
        save_txt=True,
        project=str(PREDICT_DIR),
        name=run_name,
        exist_ok=True
    )
    return results


def inference_on_single_image():
    """单张图像推理演示。"""
    print("\n" + "=" * 70)
    print("单张图像推理演示")
    print("=" * 70)

    test_dirs = ["./data/cityscapes2/images/val"]
    test_images = []
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                test_images.extend(Path(test_dir).rglob(ext))
            if test_images:
                break

    if not test_images:
        print("未找到测试图像")
        return

    test_image = str(test_images[0])
    print(f"使用测试图像: {test_image}")

    try:
        results = _common_predict(test_image, "single_image_demo")
        result = results[0]
        print(f"\n检测结果:")
        print(f"  图像尺寸: {result.orig_shape}")
        print(f"  检测到 {len(result.boxes)} 个对象")

        if result.boxes is not None:
            print("\n  检测到的对象:")
            for i, (box, conf, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)):
                class_id = int(cls)
                class_name = CLASS_TEXTS[class_id] if class_id < len(CLASS_TEXTS) else f"class_{class_id}"
                print(f"    对象 {i + 1}: {class_name}, 置信度: {conf:.3f}, 位置: {box.tolist()}")

        output_dir = PREDICT_DIR / "single_image_demo"
        if output_dir.exists():
            output_files = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
            if output_files:
                print(f"\n结果图像已保存到: {output_files[0]}")

    except Exception as e:
        print(f"单张图像推理失败: {e}")
        traceback.print_exc()


def batch_inference():
    """批量推理。"""
    print("\n" + "=" * 70)
    print("批量推理")
    print("=" * 70)

    val_dirs = ["./data/cityscapes2/images/val"]
    source_dir = next((d for d in val_dirs if os.path.exists(d)), None)

    if source_dir is None:
        print("未找到验证集图像目录")
        return

    print(f"使用验证集目录: {source_dir}")

    try:
        results = _common_predict(source_dir, "batch_inference")
        total_detections = sum(len(r.boxes) for r in results if r.boxes is not None)

        print(f"\n批量推理完成!")
        print(f"  处理图像数量: {len(results)}")
        print(f"  总检测对象数: {total_detections}")
        print(f"  结果保存到: {PREDICT_DIR / 'batch_inference'}")

    except Exception as e:
        print(f"批量推理失败: {e}")
        traceback.print_exc()


# ---- 模型导出 ----

def fix_openvino_version():
    """修复 OpenVINO 2025+ 版本的 __version__ 属性缺失问题。"""
    try:
        import openvino
        if hasattr(openvino, '__version__'):
            return openvino.__version__

        version = None

        # 方法1: get_version()
        try:
            from openvino.runtime import get_version
            version = get_version()
        except ImportError:
            pass

        # 方法2: core.get_versions()
        if version is None:
            try:
                import openvino.runtime as ov
                core = ov.Core()
                versions = core.get_versions()
                version = versions.get('IE_CORE_VERSION')
            except Exception:
                pass

        # 方法3: pkg_resources
        if version is None:
            try:
                import pkg_resources
                version = pkg_resources.get_distribution("openvino").version
            except Exception:
                pass

        # 方法4: 从路径推断
        if version is None:
            ov_path = os.path.dirname(openvino.__file__)
            for year in ['2025', '2024']:
                if year in ov_path:
                    version = f"{year}.4.1"
                    break

        if version:
            openvino.__version__ = version
            print(f"已设置 OpenVINO 版本为: {version}")
        else:
            openvino.__version__ = "2024.0.0"
            print("无法检测 OpenVINO 版本，设置为默认值: 2024.0.0")

        return openvino.__version__

    except Exception as e:
        print(f"OpenVINO 版本修复失败: {e}")
        return None


def export_model():
    """导出模型为 ONNX / TorchScript / OpenVINO 格式。"""
    print("\n" + "=" * 70)
    print("模型导出")
    print("=" * 70)

    model, model_path = _load_model_for_validation()
    if model is None:
        return

    # 统一修复 OpenVINO 版本
    fix_openvino_version()

    export_formats = [
        ("onnx", "ONNX格式"),
        ("torchscript", "TorchScript格式"),
        ("openvino", "OpenVINO格式"),
    ]

    for format_name, format_desc in export_formats:
        print(f"\n导出为 {format_desc}...")
        try:
            export_path = model.export(
                format=format_name,
                imgsz=640,
                simplify=True,
                opset=12,
                **({"workspace": 4} if format_name == "openvino" else {}),
                verbose=False
            )
            print(f"导出成功: {export_path}")
        except Exception as e:
            print(f"导出失败: {e}")
            if format_name == "openvino" and "__version__" in str(e):
                print("建议: OpenVINO 2025.x 版本与 Ultralytics 可能不完全兼容")
                print("解决方案: pip install openvino==2024.3.0 openvino-dev==2024.3.0")


# ---- 主入口 ----

def main():
    """主函数。"""
    setup_workspace()

    print("=" * 70)
    print("YOLO-World Cityscapes 验证和推理工具")
    print("=" * 70)

    print("\n请选择操作:")
    print("1. 验证模型 (评估指标)")
    print("2. 单张图像推理演示")
    print("3. 批量推理")
    print("4. 导出模型")
    print("5. 全部执行")
    print("6. 退出")

    choice = input("\n请输入选择 (1-6): ").strip()

    actions = {
        '1': validate_model,
        '2': inference_on_single_image,
        '3': batch_inference,
        '4': export_model,
        '5': lambda: (validate_model(), inference_on_single_image(), batch_inference(), export_model()),
        '6': lambda: print("退出程序"),
    }

    action = actions.get(choice)
    if action:
        action()
    else:
        print("无效选择")

    print("\n" + "=" * 70)
    print("验证和推理完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
