# benchmark_cityscapes.py
"""YOLO-World 模型性能基准测试与比较。"""

import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    CLASS_TEXTS, DEVICE, CUDA_AVAILABLE, BENCHMARK_DIR,
    BEST_PT, LAST_PT, DEFAULT_WEIGHTS, PARAM_ESTIMATES,
)
from utils import setup_workspace, get_device_name, find_model_path, load_yolo_world
from plot_utils import setup_plot_style, add_bar_labels, create_summary_textbox, save_and_close


setup_plot_style()


def benchmark_model():
    """基准测试单个模型的推理性能。"""
    print("=" * 70)
    print("YOLO-World Cityscapes 性能基准测试")
    print("=" * 70)

    # 环境检查
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {CUDA_AVAILABLE}")
        device_name = get_device_name()
        if CUDA_AVAILABLE:
            print(f"GPU: {device_name}")
    except Exception:
        print("PyTorch: 未安装")
        return False

    try:
        from ultralytics import YOLOWorld  # noqa: F401
        print("Ultralytics 导入成功")
    except ImportError:
        print("Ultralytics 导入失败")
        return False

    model_path = find_model_path(str(BEST_PT), str(LAST_PT), str(DEFAULT_WEIGHTS))
    if model_path is None:
        print("未找到模型文件")
        return False

    print(f"\n加载模型: {model_path}")
    try:
        model = load_yolo_world(model_path, CLASS_TEXTS)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False

    img_sizes = [320, 416, 512, 640, 800, 1024]
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # 预热
    print("\n预热模型...")
    for _ in range(10):
        _ = model.predict(dummy_image, imgsz=640, verbose=False)
    if CUDA_AVAILABLE:
        torch.cuda.synchronize()

    results = []
    print("\n开始基准测试...")

    for img_size in img_sizes:
        print(f"\n测试图像大小: {img_size}x{img_size}")
        test_image = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

        times = []
        memory_usages = []
        num_tests = 20

        for _ in range(num_tests):
            start_time = time.perf_counter()
            _ = model.predict(test_image, imgsz=img_size, verbose=False)

            if CUDA_AVAILABLE:
                torch.cuda.synchronize()

            elapsed = (time.perf_counter() - start_time) * 1000  # ms
            times.append(elapsed)

            if CUDA_AVAILABLE:
                memory_usages.append(torch.cuda.max_memory_allocated() / 1024 ** 2)
                torch.cuda.reset_peak_memory_stats()

        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        avg_memory = np.mean(memory_usages) if memory_usages else 0
        fps = 1000 / avg_time if avg_time > 0 else 0

        print(f"  平均推理时间: {avg_time:.2f} ms")
        print(f"  最小推理时间: {min_time:.2f} ms")
        print(f"  最大推理时间: {max_time:.2f} ms")
        print(f"  平均FPS: {fps:.2f}")
        if avg_memory > 0:
            print(f"  平均GPU内存使用: {avg_memory:.2f} MB")

        results.append({
            'image_size': img_size,
            'avg_time_ms': avg_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'fps': fps,
            'gpu_memory_mb': avg_memory,
            'device': device_name,
            'model': os.path.basename(model_path)
        })

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = str(BENCHMARK_DIR)
    os.makedirs(results_dir, exist_ok=True)

    csv_file = os.path.join(results_dir, f"benchmark_results_{timestamp}.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"\n基准测试结果已保存到: {csv_file}")

    visualize_benchmark_results(df, results_dir, timestamp)

    print("\n" + "=" * 70)
    print("基准测试总结")
    print("=" * 70)
    for result in results:
        print(f"图像大小 {result['image_size']}x{result['image_size']}: "
              f"{result['avg_time_ms']:.2f} ms, {result['fps']:.2f} FPS")

    return True


def visualize_benchmark_results(df, results_dir, timestamp):
    """可视化基准测试结果。"""
    print("\n生成基准测试可视化图表...")

    viz_dir = os.path.join(results_dir, f"visualizations_{timestamp}")
    os.makedirs(viz_dir, exist_ok=True)

    img_sizes = df['image_size'].values
    avg_times = df['avg_time_ms'].values
    fps_values = df['fps'].values
    memory_usage = df['gpu_memory_mb'].values if 'gpu_memory_mb' in df.columns else None

    # 修复：不再创建孤立的 plt.figure，直接使用 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'YOLO-World 性能基准测试 - {timestamp}', fontsize=16, fontweight='bold')

    # 1.1 推理时间曲线
    ax1 = axes[0, 0]
    ax1.plot(img_sizes, avg_times, 'o-', linewidth=3, markersize=8, color='#2E86AB')
    ax1.set_xlabel('图像大小 (像素)', fontsize=12)
    ax1.set_ylabel('推理时间 (毫秒)', fontsize=12)
    ax1.set_title('推理时间 vs 图像大小', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    for x, y in zip(img_sizes, avg_times):
        ax1.text(x, y + max(avg_times) * 0.02, f'{y:.1f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 1.2 FPS 曲线
    ax2 = axes[0, 1]
    ax2.plot(img_sizes, fps_values, 's-', linewidth=3, markersize=8, color='#A23B72')
    ax2.set_xlabel('图像大小 (像素)', fontsize=12)
    ax2.set_ylabel('FPS (帧/秒)', fontsize=12)
    ax2.set_title('FPS vs 图像大小', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    for x, y in zip(img_sizes, fps_values):
        ax2.text(x, y + max(fps_values) * 0.02, f'{y:.1f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 1.3 推理时间对比柱状图
    ax3 = axes[1, 0]
    bars3 = ax3.bar(range(len(img_sizes)), avg_times,
                    color=plt.cm.viridis(np.linspace(0.2, 0.8, len(img_sizes))))
    ax3.set_xticks(range(len(img_sizes)))
    ax3.set_xticklabels([f'{size}×{size}' for size in img_sizes], rotation=45)
    ax3.set_xlabel('图像大小', fontsize=12)
    ax3.set_ylabel('推理时间 (毫秒)', fontsize=12)
    ax3.set_title('各图像大小推理时间对比', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    add_bar_labels(ax3, bars3, avg_times, fmt="{:.1f}")

    # 1.4 性能总结
    ax4 = axes[1, 1]
    best_fps_idx = np.argmax(fps_values)
    fastest_idx = np.argmin(avg_times)

    summary_text = "性能测试总结\n\n"
    summary_text += f"测试模型: {df['model'].iloc[0]}\n"
    summary_text += f"测试设备: {df['device'].iloc[0]}\n\n"
    summary_text += "关键指标:\n"
    summary_text += f"* 最佳FPS: {fps_values[best_fps_idx]:.1f} @ {img_sizes[best_fps_idx]}×{img_sizes[best_fps_idx]}\n"
    summary_text += f"* 最快推理: {avg_times[fastest_idx]:.1f}ms @ {img_sizes[fastest_idx]}×{img_sizes[fastest_idx]}\n"
    summary_text += f"* 平均FPS: {np.mean(fps_values):.1f}\n"
    summary_text += f"* 平均推理时间: {np.mean(avg_times):.1f}ms\n"

    if memory_usage is not None and any(memory_usage):
        summary_text += f"* 平均内存使用: {np.mean(memory_usage):.1f} MB\n"
        summary_text += f"* 最大内存使用: {np.max(memory_usage):.1f} MB\n"

    summary_text += f"\n测试时间: {timestamp}"
    create_summary_textbox(ax4, summary_text, fontsize=11)

    plt.tight_layout()
    combined_chart = os.path.join(viz_dir, "benchmark_analysis.png")
    save_and_close(fig, combined_chart)

    # 2. 单独保存 FPS 图表
    fig2, ax_fps = plt.subplots(figsize=(10, 6))
    ax_fps.plot(img_sizes, fps_values, 's-', linewidth=3, markersize=10, color='#A23B72')
    ax_fps.fill_between(img_sizes, 0, fps_values, alpha=0.2, color='#A23B72')
    ax_fps.set_xlabel('图像大小 (像素)', fontsize=12)
    ax_fps.set_ylabel('FPS (帧/秒)', fontsize=12)
    ax_fps.set_title('YOLO-World FPS性能曲线', fontsize=14, fontweight='bold')
    ax_fps.grid(True, alpha=0.3)
    for x, y in zip(img_sizes, fps_values):
        ax_fps.text(x, y + max(fps_values) * 0.02, f'{y:.1f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    fps_chart = os.path.join(viz_dir, "fps_performance.png")
    save_and_close(fig2, fps_chart)

    # 3. 保存数据表格图片
    fig3, ax_table = plt.subplots(figsize=(10, 6))
    ax_table.axis('tight')
    ax_table.axis('off')

    table_data = [[f"{df['image_size'].iloc[i]}×{df['image_size'].iloc[i]}",
                   f"{df['avg_time_ms'].iloc[i]:.1f}",
                   f"{df['fps'].iloc[i]:.1f}"] for i in range(len(df))]

    table = ax_table.table(cellText=table_data,
                           colLabels=['图像大小', '推理时间(ms)', 'FPS'],
                           cellLoc='center', loc='center',
                           colColours=['#4B8BBE', '#306998', '#FFD43B'])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    plt.title('基准测试结果表', fontsize=14, fontweight='bold')

    table_chart = os.path.join(viz_dir, "benchmark_table.png")
    save_and_close(fig3, table_chart)

    print(f"所有可视化图表已生成，保存在: {viz_dir}")
    generate_html_report(df, viz_dir, timestamp)


def generate_html_report(df, viz_dir, timestamp):
    """生成 HTML 格式的基准测试报告。"""
    html_file = os.path.join(viz_dir, "benchmark_report.html")

    # 使用列表拼接代替巨大 f-string，更清晰可维护
    best_fps_idx = np.argmax(df['fps'].values)
    best_fps_size = df['image_size'].iloc[best_fps_idx]
    best_fps_value = df['fps'].iloc[best_fps_idx]

    fastest_idx = np.argmin(df['avg_time_ms'].values)
    fastest_size = df['image_size'].iloc[fastest_idx]
    fastest_time = df['avg_time_ms'].iloc[fastest_idx]

    def _memory_display(i):
        if 'gpu_memory_mb' not in df.columns:
            return 'N/A'
        val = df['gpu_memory_mb'].iloc[i]
        return 'N/A' if pd.isna(val) else f'{val:.1f}'

    table_rows = "\n".join(
        f"""                <tr>
                    <td>{df['image_size'].iloc[i]}×{df['image_size'].iloc[i]}</td>
                    <td>{df['avg_time_ms'].iloc[i]:.1f}</td>
                    <td>{df['fps'].iloc[i]:.1f}</td>
                    <td>{_memory_display(i)}</td>
                </tr>"""
        for i in range(len(df))
    )

    memory_section = ""
    if 'gpu_memory_mb' in df.columns:
        memory_values = df['gpu_memory_mb'].dropna().values
        if len(memory_values) > 0:
            memory_section = (
                f"""            <p><span class="info">[信息]</span> <strong>平均内存使用:</strong> {np.mean(memory_values):.1f} MB</p>
            <p><span class="info">[信息]</span> <strong>最大内存使用:</strong> {np.max(memory_values):.1f} MB</p>\n"""
            )
        else:
            memory_section = '            <p><span class="info">[信息]</span> <strong>内存使用数据:</strong> 不可用</p>\n'

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO-World 基准测试报告 - {timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; border-bottom: 2px solid #4B8BBE; padding-bottom: 20px; margin-bottom: 30px; }}
        .section {{ margin-bottom: 30px; padding: 20px; background: #f9f9f9; border-radius: 8px; }}
        h1 {{ color: #4B8BBE; }}
        h2 {{ color: #306998; border-left: 5px solid #4B8BBE; padding-left: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: center; border: 1px solid #ddd; }}
        th {{ background-color: #4B8BBE; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .image-container {{ text-align: center; margin: 20px 0; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
        .summary {{ background: #e7f3ff; padding: 15px; border-radius: 8px; }}
        .highlight {{ color: #FF4500; font-weight: bold; }}
        .success {{ color: green; }}
        .info {{ color: blue; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>YOLO-World 性能基准测试报告</h1>
            <p><strong>生成时间:</strong> {timestamp}</p>
        </div>
        <div class="section">
            <h2>测试概览</h2>
            <p><strong>测试模型:</strong> {df['model'].iloc[0]}</p>
            <p><strong>测试设备:</strong> {df['device'].iloc[0]}</p>
            <p><strong>测试图像大小:</strong> {', '.join(f'{size}×{size}' for size in df['image_size'].values)}</p>
        </div>
        <div class="section">
            <h2>性能数据表</h2>
            <table>
                <tr><th>图像大小</th><th>推理时间 (ms)</th><th>FPS</th><th>内存使用 (MB)</th></tr>
{table_rows}
            </table>
        </div>
        <div class="section">
            <h2>性能分析图表</h2>
            <div class="image-container"><h3>综合分析图</h3><img src="benchmark_analysis.png" alt="基准测试综合分析"></div>
            <div class="image-container"><h3>FPS性能曲线</h3><img src="fps_performance.png" alt="FPS性能曲线"></div>
            <div class="image-container"><h3>结果数据表</h3><img src="benchmark_table.png" alt="基准测试结果表"></div>
        </div>
        <div class="section summary">
            <h2>性能总结</h2>
            <p><span class="success">[成功]</span> <strong>最佳FPS性能:</strong> <span class="highlight">{best_fps_value:.1f} FPS</span> @ {best_fps_size}×{best_fps_size} 图像大小</p>
            <p><span class="info">[信息]</span> <strong>最快推理速度:</strong> <span class="highlight">{fastest_time:.1f} ms</span> @ {fastest_size}×{fastest_size} 图像大小</p>
            <p><span class="info">[信息]</span> <strong>平均FPS:</strong> {np.mean(df['fps'].values):.1f}</p>
            <p><span class="info">[信息]</span> <strong>平均推理时间:</strong> {np.mean(df['avg_time_ms'].values):.1f} ms</p>
{memory_section}        </div>
        <div class="section">
            <h2>使用建议</h2>
            <ul>
                <li>对于实时应用，推荐使用 <strong>{best_fps_size}×{best_fps_size}</strong> 分辨率以获得最佳FPS</li>
                <li>对于精度要求高的场景，可以考虑使用 <strong>640×640</strong> 分辨率</li>
                <li>内存受限环境下，建议使用较小分辨率如 <strong>320×320</strong></li>
                <li>可根据实际应用场景在速度和精度之间进行权衡</li>
            </ul>
        </div>
        <div class="section" style="text-align: center; color: #666; font-size: 0.9em;">
            <p>报告生成工具: YOLO-World 基准测试系统</p>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>"""

    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML报告已生成: {html_file}")


def compare_models():
    """比较不同模型的推理性能。"""
    print("\n" + "=" * 70)
    print("模型性能比较")
    print("=" * 70)

    models_to_compare = [
        ("yolov8s-world.pt", "YOLOv8s-World"),
        ("yolov8s-worldv2.pt", "YOLOv8s-Worldv2"),
        ("yolov8m-world.pt", "YOLOv8m-World"),
        ("yolov8m-worldv2.pt", "YOLOv8m-Worldv2"),
        ("yolov8l-world.pt", "YOLOv8l-World"),
        ("yolov8l-worldv2.pt", "YOLOv8l-Worldv2"),
        ("yolov8x-world.pt", "YOLOv8x-World"),
        ("yolov8x-worldv2.pt", "YOLOv8x-Worldv2"),
    ]

    results = []

    for model_file, model_name in models_to_compare:
        print(f"\n测试模型: {model_name}")
        model_path = f"./weights/{model_file}"

        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            continue

        try:
            model = load_yolo_world(model_path, CLASS_TEXTS)
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

            for _ in range(5):
                _ = model.predict(test_image, imgsz=640, verbose=False)
            if CUDA_AVAILABLE:
                torch.cuda.synchronize()

            times = []
            for _ in range(10):
                start_time = time.perf_counter()
                _ = model.predict(test_image, imgsz=640, verbose=False)
                if CUDA_AVAILABLE:
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start_time) * 1000)

            avg_time = np.mean(times)
            fps = 1000 / avg_time if avg_time > 0 else 0
            params = PARAM_ESTIMATES.get(model_file, 0)

            results.append({
                'model_name': model_name,
                'model_file': model_file,
                'avg_time_ms': avg_time,
                'fps': fps,
                'params_m': params,
                'device': get_device_name()
            })
            print(f"  平均推理时间: {avg_time:.2f} ms, FPS: {fps:.2f}, 参数量: {params}M")

        except Exception as e:
            print(f"测试失败: {e}")

    if not results:
        return results

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = str(BENCHMARK_DIR)
    os.makedirs(results_dir, exist_ok=True)

    csv_file = os.path.join(results_dir, f"model_comparison_{timestamp}.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"\n模型比较结果已保存到: {csv_file}")

    visualize_model_comparison(df, results_dir, timestamp)

    print("\n" + "=" * 70)
    print("模型性能比较总结")
    print("=" * 70)
    print(f"{'模型':<20} {'推理时间(ms)':<15} {'FPS':<10} {'参数量(M)':<12}")
    print("-" * 60)
    for result in results:
        print(f"{result['model_name']:<20} {result['avg_time_ms']:<15.2f} "
              f"{result['fps']:<10.2f} {result['params_m']:<12.1f}")

    return results


def visualize_model_comparison(df, results_dir, timestamp):
    """可视化模型比较结果。"""
    print("\n生成模型比较可视化图表...")

    viz_dir = os.path.join(results_dir, f"comparison_visualizations_{timestamp}")
    os.makedirs(viz_dir, exist_ok=True)

    model_names = df['model_name'].values
    avg_times = df['avg_time_ms'].values
    fps_values = df['fps'].values
    params = df['params_m'].values

    # 修复：不再创建孤立的 plt.figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'YOLO-World 模型性能比较 - {timestamp}', fontsize=16, fontweight='bold')

    # 1.1 推理时间对比柱状图
    ax1 = axes[0, 0]
    x_pos = np.arange(len(model_names))
    bars1 = ax1.bar(x_pos, avg_times, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names))))
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.set_ylabel('推理时间 (毫秒)', fontsize=12)
    ax1.set_title('模型推理时间对比', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    add_bar_labels(ax1, bars1, avg_times, fmt="{:.1f}")

    # 1.2 FPS 对比柱状图
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x_pos, fps_values, color=plt.cm.plasma(np.linspace(0.2, 0.8, len(model_names))))
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.set_ylabel('FPS (帧/秒)', fontsize=12)
    ax2.set_title('模型FPS对比', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    add_bar_labels(ax2, bars2, fps_values, fmt="{:.1f}")

    # 1.3 参数量 vs 性能散点图
    ax3 = axes[1, 0]
    scatter = ax3.scatter(params, avg_times, s=200, c=fps_values, cmap='RdYlGn_r', alpha=0.8)
    ax3.set_xlabel('参数量 (百万)', fontsize=12)
    ax3.set_ylabel('推理时间 (毫秒)', fontsize=12)
    ax3.set_title('参数量 vs 推理时间', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    for x, y, name in zip(params, avg_times, model_names):
        ax3.text(x, y, name, fontsize=9, ha='center', va='bottom')
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('FPS (帧/秒)', fontsize=12)

    # 1.4 性能排序
    ax4 = axes[1, 1]
    sorted_indices = np.argsort(fps_values)[::-1]

    summary_text = "模型性能排序\n\nFPS性能排名:\n"
    for rank, idx in enumerate(sorted_indices):
        prefix = ["第一名: ", "第二名: ", "第三名: "][rank] if rank < 3 else f"{rank + 1}. "
        summary_text += f"{prefix}{model_names[idx]}: {fps_values[idx]:.1f} FPS\n"

    summary_text += f"\n性能最佳: {model_names[sorted_indices[0]]}\n"
    summary_text += f"最快推理: {min(avg_times):.1f} ms\n"
    summary_text += f"平均FPS: {np.mean(fps_values):.1f}\n"
    summary_text += f"平均推理时间: {np.mean(avg_times):.1f} ms"

    create_summary_textbox(ax4, summary_text, fontsize=11)

    plt.tight_layout()
    combined_chart = os.path.join(viz_dir, "model_comparison_analysis.png")
    save_and_close(fig, combined_chart)

    # 2. 性能对比雷达图
    norm_fps = (fps_values - np.min(fps_values)) / (np.max(fps_values) - np.min(fps_values)) if len(fps_values) > 1 else np.ones_like(fps_values)
    norm_speed = 1 - (avg_times - np.min(avg_times)) / (np.max(avg_times) - np.min(avg_times)) if len(avg_times) > 1 else np.ones_like(avg_times)
    norm_efficiency = norm_fps / (params / np.max(params)) if np.max(params) > 0 else np.ones_like(norm_fps)

    angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
    angles += angles[:1]

    fig2, ax_radar = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
    for i, model_name in enumerate(model_names):
        values = [norm_fps[i], norm_speed[i], norm_efficiency[i]]
        values += values[:1]
        ax_radar.plot(angles, values, 'o-', linewidth=2, label=model_name)
        ax_radar.fill(angles, values, alpha=0.1)

    ax_radar.set_thetagrids(np.degrees(angles[:-1]), ['FPS性能', '推理速度', '计算效率'])
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title('模型性能雷达图', fontsize=14, fontweight='bold', pad=20)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax_radar.grid(True)

    radar_chart = os.path.join(viz_dir, "model_performance_radar.png")
    save_and_close(fig2, radar_chart)

    print(f"所有模型比较可视化图表已生成，保存在: {viz_dir}")


def visualize_existing_results():
    """可视化已存在的基准测试结果。"""
    print("\n" + "=" * 70)
    print("可视化已有基准测试结果")
    print("=" * 70)

    results_dir = str(BENCHMARK_DIR)
    if not os.path.exists(results_dir):
        print("基准测试目录不存在")
        return

    csv_files = list(Path(results_dir).glob("benchmark_results_*.csv"))
    if not csv_files:
        print("未找到基准测试结果文件")
        return

    latest_csv = max(csv_files, key=os.path.getctime)
    print(f"使用结果文件: {latest_csv.name}")

    try:
        df = pd.read_csv(latest_csv)
        print(f"读取成功，共 {len(df)} 条记录")
        timestamp = latest_csv.stem.replace("benchmark_results_", "")
        visualize_benchmark_results(df, results_dir, f"reanalysis_{timestamp}")
        print("可视化完成")
    except Exception as e:
        print(f"可视化失败: {e}")
        traceback.print_exc()


def main():
    """主函数。"""
    setup_workspace()

    print("=" * 70)
    print("YOLO-World Cityscapes 性能基准测试工具")
    print("=" * 70)

    print("\n请选择操作:")
    print("1. 单个模型基准测试")
    print("2. 多模型性能比较")
    print("3. 可视化已有结果")
    print("4. 全部执行")
    print("5. 退出")

    choice = input("\n请输入选择 (1-5): ").strip()

    actions = {
        '1': benchmark_model,
        '2': compare_models,
        '3': visualize_existing_results,
        '4': lambda: (benchmark_model(), compare_models()),
        '5': lambda: print("退出程序"),
    }

    action = actions.get(choice)
    if action:
        action()
    else:
        print("无效选择")

    print("\n" + "=" * 70)
    print("基准测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
