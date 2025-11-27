import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter


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


def plot_combined_scatter_by_metric(data, save_path=None):
    """只区分指标类型的组合散点图+边缘分布图"""

    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                           hspace=0.05, wspace=0.05)

    ax_histx = fig.add_subplot(gs[0, 0])
    ax_histy = fig.add_subplot(gs[1, 1])
    ax_scatter = fig.add_subplot(gs[1, 0])

    min_val, max_val = 0.2, 1.0
    bins = np.linspace(min_val, max_val, 21)

    all_orig = []
    all_imp = []

    # 指标对应颜色（自定义）
    metric_colors = {
        'acc': 'tomato',
        'nmi': 'dodgerblue',
        'ari': 'seagreen'
    }

    # 遍历数据（忽略方法 n，只用 metric）
    for method_data in data.values():
        for metric, data_list in method_data.items():
            orig_vals = [d['orig_val'] for d in data_list]
            imp_vals = [d['imp_val'] for d in data_list]

            metric_key = metric.lower()  # 统一小写键访问
            color = metric_colors.get(metric_key, 'gray')  # fallback 防止KeyError

            ax_scatter.scatter(orig_vals, imp_vals,
                               color=color,
                               marker='o',
                               s=80, alpha=0.7,
                               label=metric.upper())

            all_orig.extend(orig_vals)
            all_imp.extend(imp_vals)

    # 画 y = x 参考线
    ax_scatter.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, alpha=0.7)
    ax_scatter.fill_between([min_val, max_val], [min_val, max_val], [max_val, max_val],
                            color='green', alpha=0.1)

    # 改进方向箭头
    mid = (min_val + max_val) / 2
    arrow_len = 0.1 * (max_val - min_val)
    ax_scatter.annotate('', xy=(mid, mid + arrow_len / 2), xytext=(mid, mid - arrow_len / 2),
                        arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax_scatter.text(mid + 0.01, mid, '改进方向',
                    fontsize=14, color='black', ha='left', va='center')

    ax_scatter.set_xlim(min_val, max_val)
    ax_scatter.set_ylim(min_val, max_val)
    ax_scatter.set_xlabel('原方法指标值', fontsize=14)
    ax_scatter.set_ylabel('改进方法指标值', fontsize=14)
    ax_scatter.grid(True, linestyle='--', alpha=0.5)
    ax_scatter.text(0.05, 0.95, '改进方法优于原方法',
                    fontsize=14, transform=ax_scatter.transAxes,
                    bbox=dict(facecolor='white', alpha=0.7))
    ax_scatter.tick_params(labelsize=14)
    for spine in ['top', 'right']:
        ax_scatter.spines[spine].set_visible(False)

    # ==== 上方边缘分布 ====
    hist_vals_x, bin_edges_x = np.histogram(all_orig, bins=bins, weights=np.ones_like(all_orig) / len(all_orig))
    bin_centers_x = 0.5 * (bin_edges_x[:-1] + bin_edges_x[1:])
    ax_histx.bar(bin_centers_x, hist_vals_x, width=(bins[1] - bins[0]),
                 color='steelblue', alpha=0.7, edgecolor='black')
    for x, h in zip(bin_centers_x, hist_vals_x):
        ax_histx.text(x, h + 0.002, f'{h:.2f}', ha='center', va='bottom', fontsize=10)

    print(f"上方直方图频率总和: {np.sum(hist_vals_x):.4f}")

    kde_x = gaussian_kde(all_orig)
    xx = np.linspace(min_val, max_val, 200)
    ax_histx.plot(xx, kde_x(xx) * (bins[1] - bins[0]), color='darkblue', lw=2)
    ax_histx.axvline(np.mean(all_orig), color='red', linestyle='--', lw=2)

    ax_histx.tick_params(axis='x', labelbottom=False)
    ax_histx.tick_params(axis='y', labelsize=14)
    ax_histx.set_title('聚类方法改进前后指标对比', fontsize=16, pad=10)
    for spine in ['left', 'top', 'right']:
        ax_histx.spines[spine].set_visible(False)

    # ==== 右侧边缘分布 ====
    hist_vals_y, bin_edges_y = np.histogram(all_imp, bins=bins, weights=np.ones_like(all_imp) / len(all_imp))
    bin_centers_y = 0.5 * (bin_edges_y[:-1] + bin_edges_y[1:])
    ax_histy.barh(bin_centers_y, hist_vals_y, height=(bins[1] - bins[0]),
                  color='steelblue', alpha=0.7, edgecolor='black')
    for y, h in zip(bin_centers_y, hist_vals_y):
        ax_histy.text(h + 0.002, y, f'{h:.2f}', va='center', ha='left', fontsize=10)

    print(f"右侧直方图频率总和: {np.sum(hist_vals_y):.4f}")

    kde_y = gaussian_kde(all_imp)
    yy = np.linspace(min_val, max_val, 200)
    ax_histy.plot(kde_y(yy) * (bins[1] - bins[0]), yy, color='darkblue', lw=2)
    ax_histy.axhline(np.mean(all_imp), color='red', linestyle='--', lw=2)

    ax_histy.tick_params(axis='y', labelleft=False)
    ax_histy.tick_params(axis='x', labelsize=14)
    for spine in ['top', 'right', 'bottom']:
        ax_histy.spines[spine].set_visible(False)

    # 图例：指标名
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='ACC', markerfacecolor=metric_colors['acc'], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='NMI', markerfacecolor=metric_colors['nmi'], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='ARI', markerfacecolor=metric_colors['ari'], markersize=10)
    ]
    ax_scatter.legend(handles=legend_elements,
                      loc='lower right', fontsize=14, frameon=True, framealpha=0.9,
                      bbox_to_anchor=(1.0, 0.0))

    if save_path:
        plt.savefig(save_path, dpi=900, bbox_inches='tight')
        print(f"已保存组合散点图: {save_path}")

    plt.show()


# 主程序
if __name__ == "__main__":
    print(f"从文件 {excel_path} 的 {sheet_name} 中提取散点图数据...")
    all_scatter_data = extract_all_scatter_data()

    # 创建输出目录
    output_dir = '散点图对比'
    os.makedirs(output_dir, exist_ok=True)

    # 绘制并保存组合散点图
    save_path = os.path.join(output_dir, '三指标改进对比(不区分方法)+分布.png')
    print("正在生成组合散点图对比...")
    plot_combined_scatter_by_metric(all_scatter_data, save_path)

    print(f"组合散点图已生成并保存至 {output_dir} 目录")