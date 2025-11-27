import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib.lines import Line2D

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


def plot_combined_scatter_comparison(data, save_path=None):
    """绘制组合散点图对比并保存"""
    plt.figure(figsize=(12, 12))

    # 创建散点图
    for n, method_data in data.items():
        for metric, data_list in method_data.items():
            orig_vals = [d['orig_val'] for d in data_list]
            imp_vals = [d['imp_val'] for d in data_list]

            plt.scatter(orig_vals, imp_vals,
                        color=methods[n]['color'],
                        marker=metrics[metric]['marker'],
                        s=80, alpha=0.7,
                        label=None)  # 不在每个点上生成图例

    # 绘制y=x参考线
    min_val = 0.2
    max_val = 1.0  # 指标最大值为1.0
    line = np.linspace(min_val, max_val, 100)
    plt.plot(line, line, 'k--', linewidth=1.5, alpha=0.7, label='y=x')

    # 设置图表属性
    plt.xlabel('原方法指标值', fontsize=12)
    plt.ylabel('改进方法指标值', fontsize=12)
    plt.title('聚类方法改进前后指标对比', fontsize=14, pad=20)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.5)

    # 创建自定义图例（方法颜色和指标形状）
    legend_elements = []

    # 添加方法颜色图例
    for n, method_info in methods.items():
        legend_elements.append(Line2D([0], [0],
                                      marker='o',
                                      color='w',
                                      label=method_info['name'],
                                      markerfacecolor=method_info['color'],
                                      markersize=10))

    # 添加指标形状图例
    for metric, metric_info in metrics.items():
        legend_elements.append(Line2D([0], [0],
                                      marker=metric_info['marker'],
                                      color='w',
                                      label=metric_info['label'],
                                      markerfacecolor='gray',
                                      markersize=10))

    # 添加图例说明
    plt.legend(handles=legend_elements,
               loc='lower right',
               fontsize=10,
               title='图例说明:',
               title_fontsize=11,
               frameon=True,
               framealpha=0.9,
               bbox_to_anchor=(1.0, 0.0))

    # 添加性能改善区域注释
    plt.fill_between([min_val, max_val], [min_val, max_val], [max_val, max_val],
                     color='green', alpha=0.1)
    plt.text(0.05, 0.95, '改进方法优于原方法',
             fontsize=10, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.7))

    # 计算y=x线的中点位置
    mid_point = (min_val + max_val) / 2

    # 添加改进方向箭头 - 从下往上穿过y=x虚线中点
    arrow_length = 0.1 * (max_val - min_val)  # 箭头长度为坐标范围的10%

    # 箭头起点（在y=x线下方的位置）
    arrow_start_x = mid_point
    arrow_start_y = mid_point - arrow_length / 2

    # 箭头终点（在y=x线上方的位置）
    arrow_end_x = mid_point
    arrow_end_y = mid_point + arrow_length / 2

    # 绘制箭头
    plt.annotate('',
                 xy=(arrow_end_x, arrow_end_y),
                 xytext=(arrow_start_x, arrow_start_y),
                 arrowprops=dict(arrowstyle='->',
                                 #color='green',
                                 color='black',
                                 lw=2,
                                 shrinkA=0,
                                 shrinkB=0,
                                 connectionstyle='arc3,rad=0'),
                 va='center', ha='center')

    # 添加标签在箭头旁边
    plt.text(arrow_end_x + 0.02, arrow_end_y - arrow_length / 2,
             '改进方向',
             fontsize=10, color='black',
             ha='left', va='center')

    plt.tight_layout()

    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
    save_path = os.path.join(output_dir, '三指标改进对比.png')
    print("正在生成组合散点图对比...")
    plot_combined_scatter_comparison(all_scatter_data, save_path)

    print(f"组合散点图已生成并保存至 {output_dir} 目录")