import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib.ticker import FormatStrFormatter  # 导入格式化工具

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# Excel文件路径（请替换为您的实际文件路径）
excel_path = "D:\论文材料\实验结果表格.xlsx"
sheet_name = 'Sheet2'  # 所有数据都在Sheet1中

# 方法名称和颜色映射（7种方法）
methods = [
    'kmeans聚类',
    '凝聚层次聚类',
    '谱聚类',
    'GMM聚类',
    'AP聚类',
    'DP聚类',
    'FCM聚类'
]

# 简写名称映射（原方法和改进方法）
method_short_names = {
    'kmeans聚类': {'orig': 'KM', 'imp': 'T-KM'},
    '凝聚层次聚类': {'orig': 'Agg', 'imp': 'T-Agg'},
    '谱聚类': {'orig': 'Spec', 'imp': 'T-Spec'},
    'GMM聚类': {'orig': 'GMM', 'imp': 'T-GMM'},
    'AP聚类': {'orig': 'AP', 'imp': 'T-AP'},
    'DP聚类': {'orig': 'DP', 'imp': 'T-DP'},
    'FCM聚类': {'orig': 'FCM', 'imp': 'T-FCM'}
}

# 指标列映射
columns_map = {
    'Acc': {'orig_col': 'B', 'imp_col': 'C'},  # B列: 原方法Acc, C列: 改进方法Acc
    'NMI': {'orig_col': 'D', 'imp_col': 'E'},  # D列: 原方法NMI, E列: 改进方法NMI
    'ARI': {'orig_col': 'F', 'imp_col': 'G'}  # F列: 原方法ARI, G列: 改进方法ARI
}


def extract_data():
    """从Sheet1中提取所有方法的数据"""
    # 读取整个sheet
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

    # 存储所有方法的数据
    all_data = {}

    for n, method in enumerate(methods, start=1):
        # 计算行索引（Excel行号从1开始，pandas索引从0开始）
        mean_row_idx = 19 + (n - 1) * 22  # 平均值行 (第20行为索引0，所以20行→索引19)
        var_row_idx = 20 + (n - 1) * 22  # 方差行 (第21行→索引20)

        method_data = {}

        for metric, cols in columns_map.items():
            # 获取列索引
            orig_col_idx = ord(cols['orig_col']) - 65  # A=0, B=1, C=2,...
            imp_col_idx = ord(cols['imp_col']) - 65

            # 提取平均值
            orig_mean = df.iloc[mean_row_idx, orig_col_idx]
            imp_mean = df.iloc[mean_row_idx, imp_col_idx]

            # 提取方差
            orig_var = df.iloc[var_row_idx, orig_col_idx]
            imp_var = df.iloc[var_row_idx, imp_col_idx]

            # 存储到数据结构
            method_data[metric] = {
                'orig_mean': orig_mean,
                'imp_mean': imp_mean,
                'orig_var': orig_var,
                'imp_var': imp_var
            }

        all_data[method] = method_data

    return all_data

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

def plot_comparison(data, metric_name, save_path=None):
    """绘制水平条形对比图：字体统一调大为26，柱子粗细保持不变"""
    plt.rcParams['font.size'] = 28
    plt.figure(figsize=(12, 10))

    y = np.arange(len(methods))  # 默认间距
    height = 0.35  # 柱子粗细保持原样

    # 提取数据
    orig_means = [data[method][metric_name]['orig_mean'] for method in methods]
    imp_means = [data[method][metric_name]['imp_mean'] for method in methods]
    orig_vars = [data[method][metric_name]['orig_var'] for method in methods]
    imp_vars = [data[method][metric_name]['imp_var'] for method in methods]

    # 颜色
    original_color = '#87CEFA'
    improved_color = '#1E90FF'

    # 画柱子
    plt.barh(y - height / 2, orig_means, height=height,
             color=original_color, xerr=orig_vars,
             capsize=7, error_kw={'elinewidth': 1.5})

    plt.barh(y + height / 2, imp_means, height=height,
             color=improved_color, xerr=imp_vars,
             capsize=7, error_kw={'elinewidth': 1.5})

    # 坐标轴
    plt.xlabel(metric_name, fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks([])

    # 方法标签
    for i, method in enumerate(methods):
        plt.text(-0.06, y[i] - height / 2,
                 method_short_names[method]['orig'],
                 ha='center', va='center', fontsize=28,
                 transform=plt.gca().get_yaxis_transform())

        plt.text(-0.06, y[i] + height / 2,
                 method_short_names[method]['imp'],
                 ha='center', va='center', fontsize=28,
                 transform=plt.gca().get_yaxis_transform())

    # x轴设置
    x_max = max(max(orig_means + orig_vars), max(imp_means + imp_vars)) + 0.175
    plt.xlim(0.4, x_max)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # 网格线
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # 数值标签
    for i, method in enumerate(methods):
        plt.text(orig_means[i] + orig_vars[i] + 0.002,
                 y[i] - height / 2,
                 f'{orig_means[i]:.4f}±{orig_vars[i]:.4f}',
                 ha='left', va='center', fontsize=28, color='black')

        plt.text(imp_means[i] + imp_vars[i] + 0.002,
                 y[i] + height / 2,
                 f'{imp_means[i]:.4f}±{imp_vars[i]:.4f}',
                 ha='left', va='center', fontsize=28, color='black')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='eps', dpi=900, bbox_inches='tight')
        print(f"已保存图像: {save_path}")

    plt.show()



# 主程序
if __name__ == "__main__":
    print(f"从文件 {excel_path} 的 {sheet_name} 中提取数据...")
    all_data = extract_data()

    # 创建输出目录
    output_dir = '聚类方法对比图'
    os.makedirs(output_dir, exist_ok=True)

    # 绘制并保存三张图
    for metric in ['Acc', 'NMI', 'ARI']:
        # save_path = os.path.join(output_dir, f'{metric}_指标对比.png')
        save_path = os.path.join(output_dir, f'{metric}_指标对比.eps')
        print(f"正在生成 {metric} 对比图...")
        plot_comparison(all_data, metric, save_path)

    print(f"所有图表已生成并保存至 {output_dir} 目录")