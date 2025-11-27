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


def plot_delta_scatter_with_freq_line(data, save_path=None):
    """改进值 vs 原值 散点图 + 频率直方图的近似分布曲线"""

    fig, ax = plt.subplots(figsize=(10, 10))

    metric_styles = {
        'acc': {'color': 'tomato',     'marker': 'o'},
        'nmi': {'color': 'dodgerblue', 'marker': 's'},
        'ari': {'color': 'seagreen',   'marker': '^'},
    }

    all_orig_vals = []
    all_deltas = []
    metric_deltas = {k: [] for k in metric_styles}
    metric_orig = {k: [] for k in metric_styles}

    # ----------- 绘制散点图 -----------
    for method_data in data.values():
        for metric, data_list in method_data.items():
            metric_key = metric.lower()
            style = metric_styles.get(metric_key, {'color': 'gray', 'marker': 'x'})

            orig_vals = np.array([d['orig_val'] for d in data_list])
            imp_vals  = np.array([d['imp_val'] for d in data_list])
            deltas = imp_vals - orig_vals

            all_orig_vals.extend(orig_vals)
            all_deltas.extend(deltas)

            metric_deltas[metric_key].extend(deltas)
            metric_orig[metric_key].extend(orig_vals)

            ax.scatter(orig_vals, deltas,
                       color=style['color'],
                       marker=style['marker'],
                       s=80, alpha=0.7,
                       label=metric.upper())

    # ----------- 频率直方图近似曲线 -----------
    # ----------- 上边缘拟合曲线（每个 metric） -----------
    for metric_key in metric_deltas:
        x = np.array(metric_orig[metric_key])
        y = np.array(metric_deltas[metric_key])
        if len(x) < 3:
            continue

        try:
            # 1. 对 x 做分 bin（固定宽度划分）
            num_bins = 20
            bins = np.linspace(0.2, 1.0, num_bins + 1)
            bin_centers = []
            max_y_in_bin = []

            for i in range(num_bins):
                x_min, x_max = bins[i], bins[i + 1]
                mask = (x >= x_min) & (x < x_max)
                if np.sum(mask) == 0:
                    continue
                bin_x = x[mask]
                bin_y = y[mask]
                bin_centers.append((x_min + x_max) / 2)
                max_y_in_bin.append(np.max(bin_y))

            bin_centers = np.array(bin_centers)
            max_y_in_bin = np.array(max_y_in_bin)

            # 2. 拟合这些边缘点
            if len(bin_centers) >= 4:
                x_smooth = np.linspace(0, 1.0, 300)
                spline = make_interp_spline(bin_centers, max_y_in_bin, k=3)
                y_smooth = spline(x_smooth)
            else:
                x_smooth, y_smooth = bin_centers, max_y_in_bin

            ax.plot(x_smooth, y_smooth,
                    color=metric_styles[metric_key]['color'],
                    linestyle='-', linewidth=2.5,
                    label=f'{metric_key.upper()} 边缘拟合')
        except Exception as e:
            print(f"边缘拟合失败 ({metric_key}): {e}")
            continue

    # ----------- 辅助线 + 图修饰 -----------
    ax.axhline(0, color='gray', linestyle='--', lw=1)
    ax.set_xlim(0.2, 1.0)

    delta_lim = max(abs(np.min(all_deltas)), abs(np.max(all_deltas))) + 0.05
    # ax.set_ylim(-delta_lim, 1)
    ax.set_ylim(-delta_lim, 0.6)

    ax.set_xlabel('原方法指标值', fontsize=14)
    ax.set_ylabel('改进值 - 原值', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(labelsize=14)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # ----------- 图例去重 -----------
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    filtered = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    ax.legend([h for h, _ in filtered], [l for _, l in filtered],
              fontsize=13, frameon=True)

    if save_path:
        plt.savefig(save_path, dpi=900, bbox_inches='tight')
        print(f"已保存图像: {save_path}")

    plt.show()


# 主程序
if __name__ == "__main__":
    print(f"从文件 {excel_path} 的 {sheet_name} 中提取散点图数据...")
    all_scatter_data = extract_all_scatter_data()

    # 创建输出目录
    output_dir = '散点图对比'
    os.makedirs(output_dir, exist_ok=True)

    # 绘制并保存组合散点图
    save_path = os.path.join(output_dir, '三指标改进对比(不区分方法)+散点分布.png')
    print("正在生成组合散点图对比...")
    plot_delta_scatter_with_freq_line(all_scatter_data, save_path)

    print(f"组合散点图已生成并保存至 {output_dir} 目录")