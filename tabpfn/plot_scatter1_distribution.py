import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# Excel文件路径（请替换为您的实际文件路径）
excel_path = "D:\论文材料\实验结果表格.xlsx"
sheet_name = 'Sheet2'

# 方法名称和颜色映射
methods = {
    1: {'name': 'Kmeans聚类', 'color': '#17becf'},  # 新的青色
    2: {'name': '凝聚层次聚类', 'color': '#1f77b4'},  # 原有蓝色
    3: {'name': '谱聚类', 'color': '#ff7f0e'},  # 橙色
    4: {'name': 'GMM聚类', 'color': '#2ca02c'},  # 绿色
    5: {'name': 'AP聚类', 'color': '#d62728'},  # 红色
    6: {'name': 'DP聚类', 'color': '#9467bd'},  # 紫色
    7: {'name': 'FCM聚类', 'color': '#8c564b'}  # 棕色
}


def extract_scatter_data():
    """提取所有数据集的原方法和改进方法的Acc值"""
    # 读取Excel文件
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

    # 存储所有数据和点个数统计
    all_data = {}
    point_counts = {}  # 存储每个方法的点个数

    # 对于每种方法
    for n in methods.keys():
        method_data = []

        # 对于每个数据集 (1-16)
        for i in range(1, 17):
            # 计算行索引 (Excel行号从1开始，pandas索引从0开始)
            row_idx = 2 + i + 22 * (n - 1)  # 第3行对应索引2

            # 提取原方法和改进方法的Acc值
            orig_acc = df.iloc[row_idx, 1]  # B列 (索引1)
            imp_acc = df.iloc[row_idx, 2]  # C列 (索引2)

            # # 提取原方法和改进方法的NMI值
            # orig_acc = df.iloc[row_idx, 3]  # B列 (索引1)
            # imp_acc = df.iloc[row_idx, 4]  # C列 (索引2)

            # # 提取原方法和改进方法的ARI值
            # orig_acc = df.iloc[row_idx, 5]  # B列 (索引1)
            # imp_acc = df.iloc[row_idx, 6]  # C列 (索引2)

            # 仅当两个值都不为空时才添加点
            if not pd.isna(orig_acc) and not pd.isna(imp_acc):
                method_data.append({
                    'dataset': f'DS{i}',
                    'orig_acc': orig_acc,
                    'imp_acc': imp_acc
                })

        # 存储该方法的点数据
        all_data[n] = method_data

        # 统计该方法的点个数
        point_counts[methods[n]['name']] = len(method_data)

    return all_data, point_counts


def plot_scatter_comparison(data, save_path=None):
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                           hspace=0.05, wspace=0.05)

    ax_histx = fig.add_subplot(gs[0, 0])
    ax_histy = fig.add_subplot(gs[1, 1])
    ax_scatter = fig.add_subplot(gs[1, 0])

    all_orig_acc = []
    all_imp_acc = []
    for method_data in data.values():
        for d in method_data:
            all_orig_acc.append(d['orig_acc'])
            all_imp_acc.append(d['imp_acc'])

    min_val, max_val = 0.5, 1.0
    bins = np.linspace(min_val, max_val, 21)

    for n, method_data in data.items():
        orig_acc = [d['orig_acc'] for d in method_data]
        imp_acc = [d['imp_acc'] for d in method_data]
        ax_scatter.scatter(orig_acc, imp_acc,
                           color=methods[n]['color'],
                           s=80, alpha=0.7,
                           label=methods[n]['name'])

    ax_scatter.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, alpha=0.7)
    ax_scatter.fill_between([min_val, max_val], [min_val, max_val], [max_val, max_val],
                            color='green', alpha=0.1)

    mid_point = (min_val + max_val) / 2
    arrow_len = 0.1 * (max_val - min_val)
    ax_scatter.annotate('', xy=(mid_point, mid_point + arrow_len / 2),
                        xytext=(mid_point, mid_point - arrow_len / 2),
                        arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax_scatter.text(mid_point + 0.01, mid_point, '改进方向', fontsize=14, color='black')

    ax_scatter.set_xlim(min_val, max_val)
    ax_scatter.set_ylim(min_val, max_val)
    ax_scatter.set_xlabel('原方法Acc', fontsize=14)
    ax_scatter.set_ylabel('改进方法Acc', fontsize=14)
    ax_scatter.grid(True, linestyle='--', alpha=0.5)
    ax_scatter.legend(fontsize=14, loc='lower right')
    ax_scatter.text(0.05, 0.95, '改进方法优于原方法',
                    fontsize=14, transform=ax_scatter.transAxes,
                    bbox=dict(facecolor='white', alpha=0.7))
    ax_scatter.tick_params(labelsize=14)
    for spine in ['top', 'right']:
        ax_scatter.spines[spine].set_visible(False)

    # 上方直方图
    hist_vals_x, bin_edges_x = np.histogram(all_orig_acc, bins=bins,
                                            weights=np.ones_like(all_orig_acc) / len(all_orig_acc))
    bin_centers_x = 0.5 * (bin_edges_x[:-1] + bin_edges_x[1:])
    ax_histx.bar(bin_centers_x, hist_vals_x, width=(bins[1] - bins[0]),
                 color='steelblue', alpha=0.7, edgecolor='black')
    for x, h in zip(bin_centers_x, hist_vals_x):
        ax_histx.text(x, h + 0.002, f'{h:.2f}', ha='center', va='bottom', fontsize=10)

    print(f"上方直方图频率总和: {np.sum(hist_vals_x):.4f}")
    kde_x = gaussian_kde(all_orig_acc)
    xx = np.linspace(min_val, max_val, 200)
    ax_histx.plot(xx, kde_x(xx) * (bins[1] - bins[0]), color='darkblue', lw=2)
    ax_histx.axvline(np.mean(all_orig_acc), color='red', linestyle='--', lw=2, label='均值')
    ax_histx.tick_params(axis='x', labelbottom=False)
    ax_histx.tick_params(axis='y', labelsize=14)
    ax_histx.set_title('聚类方法改进前后Acc指标对比', fontsize=16, pad=10)
    for spine in ['left', 'top', 'right']:
        ax_histx.spines[spine].set_visible(False)

    # 右侧直方图
    hist_vals_y, bin_edges_y = np.histogram(all_imp_acc, bins=bins,
                                            weights=np.ones_like(all_imp_acc) / len(all_imp_acc))
    bin_centers_y = 0.5 * (bin_edges_y[:-1] + bin_edges_y[1:])
    ax_histy.barh(bin_centers_y, hist_vals_y, height=(bins[1] - bins[0]),
                  color='steelblue', alpha=0.7, edgecolor='black')
    for y, h in zip(bin_centers_y, hist_vals_y):
        ax_histy.text(h + 0.002, y, f'{h:.2f}', va='center', ha='left', fontsize=10)

    print(f"右侧直方图频率总和: {np.sum(hist_vals_y):.4f}")
    kde_y = gaussian_kde(all_imp_acc)
    yy = np.linspace(min_val, max_val, 200)
    ax_histy.plot(kde_y(yy) * (bins[1] - bins[0]), yy, color='darkblue', lw=2)
    ax_histy.axhline(np.mean(all_imp_acc), color='red', linestyle='--', lw=2)
    ax_histy.tick_params(axis='y', labelleft=False)
    ax_histy.tick_params(axis='x', labelsize=14)
    for spine in ['top', 'right', 'bottom']:
        ax_histy.spines[spine].set_visible(False)

    if save_path:
        plt.savefig(save_path, dpi=900, bbox_inches='tight')
        print(f"已保存散点图: {save_path}")

    plt.show()


# 主程序
if __name__ == "__main__":
    print(f"从文件 {excel_path} 的 {sheet_name} 中提取散点图数据...")
    scatter_data, point_counts = extract_scatter_data()

    # 输出每个方法的点个数统计
    print("\n每个方法绘制的点个数统计:")
    for method_name, count in point_counts.items():
        print(f"- {method_name}: {count}个点")

    # 创建输出目录
    output_dir = '散点图对比'
    os.makedirs(output_dir, exist_ok=True)

    # 绘制并保存散点图
    save_path = os.path.join(output_dir, 'Acc指标改进对比+分布.png')
    # save_path = os.path.join(output_dir, 'NMI指标改进对比+分布.png')
    # save_path = os.path.join(output_dir, 'ARI指标改进对比+分布.png')
    print("\n正在生成散点图对比...")
    plot_scatter_comparison(scatter_data, save_path)

    print(f"\n散点图已生成并保存至 {output_dir} 目录")