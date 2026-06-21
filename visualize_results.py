# visualize_results.py
"""YOLO-World 训练 / 验证 / 预测结果的可视化分析。"""

import os
import sys
import json
import re
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import CLASS_TEXTS, TRAIN_DIR, VAL_DIR, PREDICT_DIR
from utils import setup_workspace, find_latest_experiment_dir
from plot_utils import setup_plot_style, add_bar_labels, create_summary_textbox, save_and_close, build_radar_chart


setup_plot_style()


# ---- 训练结果 ----

def visualize_training_results():
    """可视化训练结果 — 仅依赖 results.csv。"""
    print("=" * 70)
    print("训练结果可视化 (仅使用results.csv)")
    print("=" * 70)

    train_dirs = [str(TRAIN_DIR), str(TRAIN_DIR.parent)]
    train_dir = None
    for dir_path in train_dirs:
        if os.path.exists(dir_path):
            latest = find_latest_experiment_dir(dir_path)
            if latest:
                train_dir = latest
                break

    if train_dir is None:
        print("未找到训练结果目录")
        return

    print(f"使用训练目录: {train_dir}")

    viz_dir = Path(train_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    results_file = Path(train_dir) / "results.csv"
    if not results_file.exists():
        print(f"未找到结果文件: {results_file}")
        print("请确保训练已完成并生成了results.csv文件")
        for item in sorted(Path(train_dir).iterdir())[:10]:
            print(f"  - {item.name}")
        return

    print(f"找到结果文件: {results_file}")

    try:
        df = pd.read_csv(results_file)
        print(f"读取成功，共 {len(df)} 轮训练数据")
        print(f"可用指标: {list(df.columns)}")

        if 'epoch' not in df.columns:
            print("未找到'epoch'列，将使用索引作为训练轮次")
            df['epoch'] = df.index + 1

        # 检测可用指标
        metric_pairs = [
            ('metrics/mAP50(B)', 'mAP50'),
            ('metrics/mAP50-95(B)', 'mAP50-95'),
            ('metrics/precision(B)', '精确率'),
            ('metrics/recall(B)', '召回率'),
            ('fitness', '适应度'),
        ]

        available = [(col, title) for col, title in metric_pairs if col in df.columns]
        if not available:
            print("结果文件中没有找到有效的指标列")
            return

        n_metrics = len(available)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, ((col, title), ax) in enumerate(zip(available, axes)):
            ax.plot(df['epoch'], df[col], linewidth=2, color=plt.cm.tab10(i % 10))
            ax.set_xlabel('训练轮次')
            ax.set_ylabel('分数')
            ax.set_title(f'{title}曲线')
            ax.grid(True, alpha=0.3)
            if len(df) == 1:
                ax.scatter(df['epoch'], df[col], color='red', s=100, zorder=5)
                ax.text(df['epoch'].iloc[0], df[col].iloc[0],
                        f'{df[col].iloc[0]:.3f}', ha='center', va='bottom',
                        fontsize=10, fontweight='bold')

        # 隐藏多余子图
        for j in range(len(available), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        results_plot = viz_dir / "training_metrics_summary.png"
        save_and_close(fig, str(results_plot), dpi=300)
        print(f"训练指标汇总图已保存: {results_plot}")

        # 生成训练报告
        _write_training_summary(viz_dir, train_dir, df, available)

        # 保存 CSV
        stats_csv = viz_dir / "training_metrics.csv"
        df.to_csv(stats_csv, index=False, encoding='utf-8')
        print(f"训练指标表格已保存: {stats_csv}")

        # 尝试打开图表
        try:
            os.startfile(str(results_plot)) if sys.platform == "win32" else None
            print(f"已为您打开训练指标汇总图")
        except Exception:
            pass

        print(f"\n训练结果可视化完成!")

    except Exception as e:
        print(f"处理训练结果失败: {e}")
        traceback.print_exc()


def _write_training_summary(viz_dir, train_dir, df, available):
    """写入训练摘要文本文件。"""
    summary_file = viz_dir / "training_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("YOLO-World Cityscapes 训练详细报告\n")
        f.write("=" * 70 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"训练目录: {train_dir}\n")
        f.write(f"训练轮次: {len(df)}\n")
        f.write(f"可用指标: {', '.join(c for c, _ in available)}\n\n")

        f.write("【关键训练指标】\n")
        f.write("-" * 40 + "\n")
        for col, title in available:
            if len(df) > 1:
                best_epoch = df[col].idxmax()
                best_value = df.loc[best_epoch, col]
                best_epoch_num = df.loc[best_epoch, 'epoch']
                f.write(f"最佳{title}: {best_value:.4f} (第{int(best_epoch_num)}轮)\n")
            else:
                f.write(f"{title}: {df[col].iloc[0]:.4f} (第1轮)\n")

        f.write("\n【数据统计】\n")
        f.write("-" * 40 + "\n")
        for col, title in available:
            if len(df) > 1:
                f.write(f"{title:10s}: 均值={df[col].mean():.4f}, "
                        f"标准差={df[col].std():.4f}, "
                        f"范围=[{df[col].min():.4f}, {df[col].max():.4f}]\n")
            else:
                f.write(f"{title:10s}: 值={df[col].iloc[0]:.4f}\n")

        f.write("\n【生成文件】\n")
        f.write("-" * 40 + "\n")
        f.write("1. 训练指标汇总图: training_metrics_summary.png\n")
        f.write("2. 本详细报告: training_summary.txt\n")

        weights_dir = Path(train_dir) / "weights"
        if weights_dir.exists():
            weight_files = []
            for ext in ['.pt', '.pth']:
                weight_files.extend(weights_dir.glob(f"*{ext}"))
            if weight_files:
                f.write("\n【模型权重】\n")
                f.write("-" * 40 + "\n")
                for wf in sorted(weight_files)[:3]:
                    f.write(f"  • {wf.name} ({wf.stat().st_size / (1024*1024):.1f} MB)\n")

    print(f"训练详细报告已保存: {summary_file}")


# ---- 验证结果 ----

def visualize_validation_results():
    """可视化验证结果。"""
    print("\n" + "=" * 70)
    print("验证结果可视化")
    print("=" * 70)

    val_dirs = [str(VAL_DIR)]
    val_dir = None
    for base_dir in val_dirs:
        if os.path.exists(base_dir):
            latest_val = find_latest_experiment_dir(base_dir, r"(val|exp)\d*$")
            if latest_val:
                val_dir = latest_val
                break

    if val_dir is None:
        print("未找到验证结果目录")
        print("建议: 请先运行 validate_cityscapes.py 进行模型验证")
        return

    print(f"使用验证目录: {val_dir}")

    # 查找 JSON 文件（按修改时间排序取最新）
    json_files = []
    for root, dirs, files in os.walk(val_dir):
        for file in files:
            if file.endswith('.json') and 'metrics' in file.lower():
                json_files.append(os.path.join(root, file))

    image_files = []
    for ext in ['.png', '.jpg', '.jpeg']:
        image_files.extend(Path(val_dir).rglob(f"*{ext}"))

    if not json_files and not image_files:
        print("未找到验证指标文件(.json)或图片文件")
        for item in sorted(os.listdir(val_dir))[:10]:
            print(f"  - {item}")
        return

    viz_dir = Path(val_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    if json_files:
        # 按时间排序取最新
        latest_json = max(json_files, key=os.path.getctime)
        print(f"使用结果文件: {os.path.basename(latest_json)}")
        try:
            with open(latest_json, 'r', encoding='utf-8') as f:
                results = json.load(f)
            generate_validation_charts_from_json(results, viz_dir)
        except Exception as e:
            print(f"读取JSON文件失败: {e}")

    if image_files:
        print(f"\n找到 {len(image_files)} 个验证图片文件")
        images_dir = viz_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        copied_images = []
        for img_file in image_files[:20]:
            dest_path = images_dir / img_file.name
            if not dest_path.exists():
                import shutil
                shutil.copy2(str(img_file), str(dest_path))
                copied_images.append(img_file.name)
        if copied_images:
            print(f"已复制 {len(copied_images)} 个图片到可视化目录")

    generate_validation_summary(val_dir, viz_dir, json_files, image_files)
    print("\n验证结果可视化完成!")


def generate_validation_charts_from_json(results, viz_dir):
    """从 JSON 结果生成验证图表。"""
    try:
        if 'box' not in results:
            print("JSON文件中未找到检测指标")
            return

        box = results['box']
        print("\n从JSON数据生成验证图表...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('YOLO-World Cityscapes 验证结果分析', fontsize=16, fontweight='bold')

        # 1. mAP 指标柱状图
        ax1 = axes[0, 0]
        map_metrics, map_labels = [], []
        for key, label in [('map50', 'mAP50'), ('map75', 'mAP75'), ('map', 'mAP50-95')]:
            if key in box:
                map_metrics.append(box[key])
                map_labels.append(label)

        if map_metrics:
            colors = ['#2E86AB', '#A23B72', '#F18F01'][:len(map_metrics)]
            bars1 = ax1.bar(map_labels, map_metrics, color=colors)
            ax1.set_ylabel('分数')
            ax1.set_title('mAP指标对比')
            ax1.grid(True, alpha=0.3, axis='y')
            add_bar_labels(ax1, bars1, map_metrics, fmt="{:.3f}")

        # 2. 各类别 AP 条形图
        ax2 = axes[0, 1]
        if 'maps' in box and isinstance(box['maps'], list):
            class_maps = box['maps']
            top_n = min(15, len(class_maps))
            top_indices = np.argsort(class_maps)[-top_n:][::-1]

            top_classes, top_values = [], []
            for idx in top_indices:
                top_values.append(class_maps[idx])
                top_classes.append(CLASS_TEXTS[idx] if idx < len(CLASS_TEXTS) else f'类别{idx}')

            colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(top_values)))
            bars2 = ax2.barh(range(len(top_values)), top_values, color=colors)
            ax2.set_yticks(range(len(top_values)))
            ax2.set_yticklabels(top_classes, fontsize=9)
            ax2.set_xlabel('AP分数')
            ax2.set_title(f'Top {top_n} 类别AP')
            ax2.grid(True, alpha=0.3, axis='x')
            for i, value in enumerate(top_values):
                ax2.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=8)

        # 3. 性能指标总结
        ax3 = axes[1, 0]
        summary_text = "验证结果总结\n\n"
        for key, label in [('map', 'mAP50-95'), ('map50', 'mAP50'), ('map75', 'mAP75')]:
            if key in box:
                summary_text += f"{label}: {box[key]:.4f}\n"
        for key, label in [('precision', '精确率'), ('recall', '召回率')]:
            if key in box and box[key] is not None:
                val = np.mean(box[key]) if isinstance(box[key], list) else box[key]
                summary_text += f"{label}: {val:.4f}\n"

        if 'maps' in box and isinstance(box['maps'], list) and box['maps']:
            class_maps = box['maps']
            best_idx = np.argmax(class_maps)
            worst_idx = np.argmin(class_maps)
            best_name = CLASS_TEXTS[best_idx] if best_idx < len(CLASS_TEXTS) else f'类别{best_idx}'
            worst_name = CLASS_TEXTS[worst_idx] if worst_idx < len(CLASS_TEXTS) else f'类别{worst_idx}'
            summary_text += f"\n最佳类别: {best_name} (AP={class_maps[best_idx]:.4f})\n"
            summary_text += f"最差类别: {worst_name} (AP={class_maps[worst_idx]:.4f})"

        create_summary_textbox(ax3, summary_text, fontsize=11)
        ax3.set_title('性能指标总结')

        # 4. 精确率-召回率雷达图
        ax4 = axes[1, 1]
        if 'precision' in box and 'recall' in box:
            precision = np.mean(box['precision']) if isinstance(box['precision'], list) else box['precision']
            recall = np.mean(box['recall']) if isinstance(box['recall'], list) else box['recall']
            build_radar_chart(ax4, [precision, recall], ['精确率', '召回率'], '精确率-召回率雷达图', '#2E8B57')
        else:
            ax4.axis('off')
            ax4.text(0.5, 0.5, '无精确率/召回率数据', ha='center', va='center', fontsize=12)

        plt.tight_layout()
        chart_path = viz_dir / "validation_analysis.png"
        save_and_close(fig, str(chart_path), dpi=300)
        print(f"验证分析图表已保存: {chart_path}")

    except Exception as e:
        print(f"生成验证图表失败: {e}")
        traceback.print_exc()


def generate_validation_summary(val_dir, viz_dir, json_files, image_files):
    """生成验证总结报告。"""
    summary_file = viz_dir / "validation_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("YOLO-World Cityscapes 验证结果总结\n")
        f.write("=" * 70 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"验证目录: {val_dir}\n")
        f.write(f"找到的JSON文件: {len(json_files)} 个\n")
        f.write(f"找到的图片文件: {len(image_files)} 个\n\n")

        if json_files:
            f.write("【指标文件】\n")
            f.write("-" * 40 + "\n")
            for jf in sorted(json_files)[:5]:
                f.write(f"• {os.path.basename(jf)}\n")
            if len(json_files) > 5:
                f.write(f"• ... 还有 {len(json_files) - 5} 个文件\n")

        if image_files:
            f.write("\n【可视化文件】\n")
            f.write("-" * 40 + "\n")
            important_patterns = ['validation_metrics_comprehensive', 'mAP_comparison',
                                  'confusion_matrix', 'PR_curve', 'val_batch']
            important = []
            other = []
            for img in image_files:
                img_name = img.name
                if any(p in img_name for p in important_patterns):
                    important.append(img_name)
                else:
                    other.append(img_name)

            for img in sorted(set(important))[:10]:
                f.write(f"• {img}\n")
            if other:
                f.write(f"\n其他图片 ({len(other)} 个):\n")
                for img in sorted(set(other))[:5]:
                    f.write(f"  - {img}\n")

        f.write("\n【使用说明】\n")
        f.write("-" * 40 + "\n")
        f.write("1. 关键图表: validation_analysis.png (综合性能分析)\n")
        f.write("2. 原始验证结果: 查看 images/ 目录下的原始图片\n")
        f.write("3. 如需重新验证: 运行 validate_cityscapes.py\n")
        f.write("4. 查看详细训练曲线: 运行本脚本选择训练结果可视化\n")

    print(f"验证总结报告已保存: {summary_file}")


# ---- 预测结果 ----

def visualize_predictions():
    """可视化预测结果。"""
    print("\n" + "=" * 70)
    print("预测结果可视化")
    print("=" * 70)

    predict_dirs = [
        str(PREDICT_DIR / "batch_inference"),
        str(PREDICT_DIR / "single_image_demo"),
        str(PREDICT_DIR),
    ]

    predict_dir = next((d for d in predict_dirs if os.path.exists(d)), None)
    if predict_dir is None:
        print("未找到预测结果目录")
        return

    print(f"使用预测目录: {predict_dir}")

    image_files = []
    label_files = []
    for root, dirs, files in os.walk(predict_dir):
        for file in files:
            path = os.path.join(root, file)
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(path)
            elif file.endswith('.txt'):
                label_files.append(path)

    if not image_files:
        print("未找到预测图像")
        return

    print(f"找到 {len(image_files)} 张预测图像，{len(label_files)} 个标签文件")

    viz_dir = Path(predict_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 分析标签统计
    class_counts = {i: 0 for i in range(len(CLASS_TEXTS))}
    detection_stats = []
    total_detections = 0

    for label_file in label_files[:100]:
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_id = int(parts[0])
                    confidence = float(parts[5]) if len(parts) > 5 else 1.0
                    if class_id in class_counts:
                        class_counts[class_id] += 1
                        total_detections += 1
                        detection_stats.append({
                            'class_id': class_id,
                            'class_name': CLASS_TEXTS[class_id] if class_id < len(CLASS_TEXTS) else f'class_{class_id}',
                            'confidence': confidence
                        })
        except Exception:
            pass

    if not detection_stats:
        print("未找到预测统计数据")
        return

    df_stats = pd.DataFrame(detection_stats)

    # 1. 类别分布图
    print("生成类别分布图...")
    class_dist = df_stats['class_name'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    bars = ax1.bar(range(len(class_dist)), class_dist.values,
                   tick_label=class_dist.index)
    ax1.set_xlabel('类别')
    ax1.set_ylabel('检测数量')
    ax1.set_title('检测类别分布')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    for i, val in enumerate(class_dist.values):
        ax1.text(i, val + 0.5, str(val), ha='center', va='bottom')

    class_dist_plot = viz_dir / "class_distribution.png"
    save_and_close(fig1, str(class_dist_plot), dpi=300)
    print(f"类别分布图已保存: {class_dist_plot}")

    # 2. 置信度分布图
    print("生成置信度分布图...")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(df_stats['confidence'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax2.set_xlabel('置信度')
    ax2.set_ylabel('频率')
    ax2.set_title('检测置信度分布')
    ax2.grid(True, alpha=0.3)

    conf_plot = viz_dir / "confidence_distribution.png"
    save_and_close(fig2, str(conf_plot), dpi=300)
    print(f"置信度分布图已保存: {conf_plot}")

    # 3. 总结报告
    summary_file = viz_dir / "prediction_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("YOLO-World Cityscapes 预测总结报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"预测目录: {predict_dir}\n")
        f.write(f"分析图像数量: {len(image_files)}\n")
        f.write(f"总检测数量: {total_detections}\n")
        f.write(f"平均每图像检测数: {total_detections / len(image_files):.2f}\n\n")
        f.write("检测类别统计:\n")
        for class_id in sorted(class_counts.keys()):
            if class_counts[class_id] > 0:
                f.write(f"  {CLASS_TEXTS[class_id]}: {class_counts[class_id]} 次检测\n")

    print(f"预测总结报告已保存: {summary_file}")

    stats_csv = viz_dir / "detection_statistics.csv"
    df_stats.to_csv(stats_csv, index=False, encoding='utf-8')
    print(f"检测统计表格已保存: {stats_csv}")
    print("\n预测结果可视化完成!")


# ---- 主入口 ----

def main():
    """主函数。"""
    setup_workspace()

    print("=" * 70)
    print("YOLO-World Cityscapes 智能可视化分析工具")
    print("=" * 70)

    print(f"\n系统状态:")
    print(f"  工作目录: {os.getcwd()}")
    print(f"  当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\n目录检查:")
    for dir_path in ["./runs/train", "./runs/val", "./runs/predict"]:
        if os.path.exists(dir_path):
            subdirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
            print(f"  ✓ {dir_path}: {len(subdirs)} 个实验")
        else:
            print(f"  ✗ {dir_path}: 不存在")

    print("\n" + "=" * 70)
    print("请选择分析操作:")
    print("=" * 70)
    print("1. 可视化训练结果")
    print("2. 可视化验证结果")
    print("3. 可视化预测结果")
    print("4. 生成完整分析报告")
    print("5. 退出程序")

    choice = input("\n请输入选择 (1-5): ").strip()

    actions = {
        '1': visualize_training_results,
        '2': visualize_validation_results,
        '3': visualize_predictions,
        '4': lambda: (visualize_training_results(), visualize_validation_results(), visualize_predictions()),
        '5': lambda: print("退出程序"),
    }

    action = actions.get(choice)
    if action:
        action()
    else:
        print("无效选择，请输入1-5之间的数字")

    print("\n" + "=" * 70)
    print("分析完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
