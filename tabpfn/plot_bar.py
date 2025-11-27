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


def plot_comparison(data, metric_name, save_path=None):
    """绘制对比图并保存"""
    plt.figure(figsize=(16, 8))  # 增加宽度以适应7种方法
    x = np.arange(len(methods))
    width = 0.35

    # 准备数据
    orig_means = [data[method][metric_name]['orig_mean'] for method in methods]
    imp_means = [data[method][metric_name]['imp_mean'] for method in methods]
    orig_vars = [data[method][metric_name]['orig_var'] for method in methods]
    imp_vars = [data[method][metric_name]['imp_var'] for method in methods]

    # 设置蓝色系颜色
    original_color = '#87CEFA'  # 浅蓝色 (原方法)
    improved_color = '#1E90FF'  # 深蓝色 (改进方法)

    # 绘制柱状图 - 原方法
    rects1 = plt.bar(x - width / 2, orig_means, width,
                     color=original_color,
                     yerr=orig_vars, capsize=7, error_kw={'elinewidth': 1.5})

    # 绘制柱状图 - 改进方法
    rects2 = plt.bar(x + width / 2, imp_means, width,
                     color=improved_color,
                     yerr=imp_vars, capsize=7, error_kw={'elinewidth': 1.5})

    # 设置图表属性
    plt.ylabel(metric_name, fontsize=16)
    plt.title(f'聚类方法{metric_name}指标对比', fontsize=16, pad=20)

    # 不设置x轴刻度标签
    plt.xticks([])  # 移除所有x轴刻度标签

    # 设置x轴刻度位置（每个方法组的位置）
    # plt.xticks(x, [method_short_names[m]['orig'] for m in methods],
    #            fontsize=11, rotation=0)  # 水平显示方法名称

    # 添加每个柱子的具体标签（在柱子正下方）
    for i, method in enumerate(methods):
        # 原方法柱子标签
        plt.text(x[i] - width / 2, -0.02,
                 method_short_names[method]['orig'],
                 ha='center', va='top', fontsize=16,
                 transform=plt.gca().get_xaxis_transform())

        # 改进方法柱子标签
        plt.text(x[i] + width / 2, -0.02,
                 method_short_names[method]['imp'],
                 ha='center', va='top', fontsize=16,
                 transform=plt.gca().get_xaxis_transform())

    # 自动设置Y轴范围(确保包含所有数据)
    #y_max = max(max(orig_means), max(imp_means)) * 1.2
    y_max = max(max(orig_means), max(imp_means)) * 1.1
    plt.ylim(0.45, y_max)

    # 保证所有y轴刻度显示两位小数
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.yticks(fontsize=18)

    #plt.legend(fontsize=11, loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 添加柱顶数值标签 - 保留4位小数
    for rects, color in zip([rects1, rects2], [original_color, improved_color]):
        for rect in rects:
            height = rect.get_height()
            # 使用:.4f格式保留4位小数
            plt.annotate(f'{height:.4f}',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=16,
                         color='black')

    plt.tight_layout()

    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=900, bbox_inches='tight')
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
        save_path = os.path.join(output_dir, f'{metric}_指标对比.png')
        print(f"正在生成 {metric} 对比图...")
        plot_comparison(all_data, metric, save_path)

    print(f"所有图表已生成并保存至 {output_dir} 目录")