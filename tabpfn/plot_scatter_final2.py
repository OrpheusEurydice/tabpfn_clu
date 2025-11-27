import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from scipy.interpolate import make_interp_spline

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# Excel文件路径（请替换为您的实际文件路径）
excel_path = "D:\论文材料\实验结果表格.xlsx"
sheet_name = 'Sheet2'

# 方法名称和颜色映射（7种方法）
methods = {
    1: {'name': 'Kmeans聚类', 'color': '#17becf'},  # 青色
    2: {'name': '凝聚层次聚类', 'color': '#1f77b4'},  # 蓝色
    3: {'name': '谱聚类', 'color': '#ff7f0e'},  # 橙色
    4: {'name': 'GMM聚类', 'color': '#2ca02c'},  # 绿色
    5: {'name': 'AP聚类', 'color': '#d62728'},  # 红色
    6: {'name': 'DP聚类', 'color': '#9467bd'},  # 紫色
    7: {'name': 'FCM聚类', 'color': '#8c564b'}  # 棕色
}

# 指标名称和形状映射
metrics = {
    'Acc': {'marker': 'o', 'label': 'Acc'},  # 圆形
    'NMI': {'marker': 's', 'label': 'NMI'},  # 方形
    'ARI': {'marker': '^', 'label': 'ARI'}  # 三角形
}



def extract_all_scatter_data():
    """提取所有数据集的原方法和改进方法的指标值"""
    # 读取Excel文件
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

    # 存储所有数据
    all_data = {}

    # 对于每种方法
    for n in methods.keys():
        method_data = {}

        # 对于每个指标
        for metric in metrics.keys():
            metric_data = []

            # 对于每个数据集 (1-16)
            for i in range(1, 17):
                # 计算行索引 (Excel行号从1开始，pandas索引从0开始)
                row_idx = 2 + i + 22 * (n - 1)  # 第3行对应索引2

                # 根据指标确定列索引
                if metric == 'Acc':
                    orig_col_idx = 1  # B列 (索引1)
                    imp_col_idx = 2  # C列 (索引2)
                elif metric == 'NMI':
                    orig_col_idx = 3  # D列 (索引3)
                    imp_col_idx = 4  # E列 (索引4)
                elif metric == 'ARI':
                    orig_col_idx = 5  # F列 (索引5)
                    imp_col_idx = 6  # G列 (索引6)

                # 提取原方法和改进方法的指标值
                orig_val = df.iloc[row_idx, orig_col_idx]
                imp_val = df.iloc[row_idx, imp_col_idx]

                # 仅当两个值都不为空时才添加点
                if not pd.isna(orig_val) and not pd.isna(imp_val):
                    metric_data.append({
                        'dataset': f'DS{i}',
                        'orig_val': orig_val,
                        'imp_val': imp_val
                    })

            method_data[metric] = metric_data

        all_data[n] = method_data

    return all_data



def plot_each_metric_separately(data, save_dir='散点图对比'):
    """每种指标绘制一张图（散点图 + 上边缘拟合曲线），并分别保存"""

    # 聚合数据（以全局 metrics 为准）
    metric_deltas = {k: [] for k in metrics}
    metric_orig = {k: [] for k in metrics}

    for method_data in data.values():
        for metric, data_list in method_data.items():
            if metric not in metrics:
                continue

            orig_vals = np.array([d['orig_val'] for d in data_list])
            imp_vals = np.array([d['imp_val'] for d in data_list])
            deltas = imp_vals - orig_vals

            metric_deltas[metric].extend(deltas)
            metric_orig[metric].extend(orig_vals)

    # 遍历每种指标，单独画图
    for metric in metrics:
        x = np.array(metric_orig[metric])
        y = np.array(metric_deltas[metric])
        if len(x) < 3:
            continue

        fig, ax = plt.subplots(figsize=(8, 8))

        marker = metrics[metric]['marker']
        label_name = metrics[metric]['label']
        color_map = {
            'Acc': 'tomato',
            'NMI': 'dodgerblue',
            'ARI': 'seagreen'
        }
        color = color_map.get(metric, 'gray')

        ax.scatter(x, y,
                   color=color,
                   marker=marker,
                   s=80, alpha=0.7,
                   label=f'{label_name}')

        # 上边缘拟合曲线（加权拟合）
        try:
            weights = np.maximum(y, 0)
            if np.sum(weights) == 0:
                continue

            x_anchor = [1.0]
            y_anchor = [0.0]
            w_anchor = [3.0]

            if metric == 'Acc':
                x_anchor.insert(0, 0.5)
                y_anchor.insert(0, 0.0)
                w_anchor.insert(0, 3.0)

            x_aug = np.concatenate([x, x_anchor])
            y_aug = np.concatenate([y, y_anchor])
            w_aug = np.concatenate([weights, w_anchor])

            deg = 4
            V = np.vander(x_aug, deg + 1)
            W = np.diag(w_aug)
            beta = np.linalg.pinv(V.T @ W @ V) @ (V.T @ W @ y_aug)
            poly = np.poly1d(beta)

            x_smooth = np.linspace(0.2, 1.0, 300)
            y_smooth = poly(x_smooth)
            y_smooth = np.clip(y_smooth, 0, None)
        except Exception as e:
            print(f"{metric} 加权拟合失败: {e}")
            continue

        from itertools import groupby
        from operator import itemgetter

        mask = y_smooth >= 0
        groups = []
        for k, g in groupby(enumerate(mask), key=lambda t: t[1]):
            if k:
                indices = list(map(itemgetter(0), g))
                groups.append(indices)

        for idx_group in groups:
            xs = x_smooth[idx_group]
            ys = y_smooth[idx_group]
            ax.plot(xs, ys,
                    color=color,
                    linestyle='-', linewidth=2.5,
                    label=f'{label_name} 加权拟合曲线' if idx_group == groups[0] else None)

        ax.axhline(0, color='gray', linestyle='--', lw=1)

        if metric == 'Acc':
            ax.set_xlim(0.5, 1.0)
        else:
            ax.set_xlim(0.2, 1.0)

        # ax.set_xlim(0.2, 1.0)
        ax.set_ylim(-0.5, 0.6)

        # 统一设置字体大小为 26
        ax.set_xlabel('原方法指标值', fontsize=28, labelpad=8)
        ax.set_ylabel('改进值 - 原值', fontsize=28, labelpad=8)
        ax.tick_params(labelsize=28)

        ax.grid(True, linestyle='--', alpha=0.4)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        ax.legend(fontsize=26, frameon=True, loc='best', framealpha=0.9)

        fig.tight_layout(pad=0.5)
        plt.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.12)

        os.makedirs(save_dir, exist_ok=True)
        # filename = f"{label_name}指标改进对比(不区分方法)+散点分布.png"
        filename = f"{label_name}指标改进对比(不区分方法)+散点分布.eps"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, format='eps', dpi=900, bbox_inches='tight', pad_inches=0.05)
        print(f"已保存图像: {save_path}")
        plt.close()


# 主程序
if __name__ == "__main__":
    print(f"从文件 {excel_path} 的 {sheet_name} 中提取散点图数据...")
    all_scatter_data = extract_all_scatter_data()

    # 创建输出目录
    output_dir = '散点图对比'
    os.makedirs(output_dir, exist_ok=True)

    # 绘制并保存每个指标单独的散点图
    print("正在生成每个指标的单独散点图对比...")
    plot_each_metric_separately(all_scatter_data, save_dir=output_dir)

    print(f"所有指标的图已生成并保存至 {output_dir} 目录")
