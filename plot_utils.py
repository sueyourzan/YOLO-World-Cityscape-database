# plot_utils.py - YOLO-World 项目共享绘图配置
"""统一的 matplotlib 样式与可复用绘图组件。"""

import matplotlib.pyplot as plt
import numpy as np


def setup_plot_style():
    """设置全局 matplotlib 样式（中文字体等），所有绘图脚本调用一次即可。"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def add_bar_labels(ax, bars, values, fmt: str = "{:.1f}", offset_ratio: float = 0.02,
                   fontsize: int = 9, fontweight: str = "bold"):
    """在柱状图上统一添加数值标签。"""
    max_val = max(values) if values else 1
    for bar, value in zip(bars, values):
        if ax.name == 'polar':
            continue
        try:
            height = bar.get_height()
        except AttributeError:
            height = bar.get_width()
        ax.text(bar.get_x() + bar.get_width() / 2, height + max_val * offset_ratio,
                fmt.format(value), ha='center', va='bottom',
                fontsize=fontsize, fontweight=fontweight)


def create_summary_textbox(ax, text: str, fontsize: int = 11,
                           facecolor: str = 'lightyellow', edgecolor: str = 'orange'):
    """在指定的 axes 上创建一个带样式的总结文本框。"""
    ax.axis('off')
    ax.text(0.1, 0.5, text, fontsize=fontsize, va='center',
            bbox=dict(boxstyle='round', facecolor=facecolor, alpha=0.8,
                      edgecolor=edgecolor, linewidth=2))


def save_and_close(fig, filepath: str, dpi: int = 150):
    """保存图表并关闭，防止内存泄漏。"""
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def build_radar_chart(ax, values, labels, title: str = "", color: str = '#2E86AB'):
    """
    在 polar axes 上绘制雷达图。

    Args:
        ax: matplotlib polar axes
        values: 各维度的值列表
        labels: 各维度的标签列表
        title: 图表标题
        color: 线条和填充颜色
    """
    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False).tolist()
    values = list(values) + [values[0]]
    angles = angles + [angles[0]]

    ax.plot(angles, values, 'o-', linewidth=2, color=color)
    ax.fill(angles, values, alpha=0.25, color=color)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)
    ax.set_ylim(0, 1.0)
