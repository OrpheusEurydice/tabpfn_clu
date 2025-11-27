import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import scipy.stats as stats

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

# 指标映射（列索引）
metrics = {
    'ACC': {'orig_col': 1, 'imp_col': 2},  # B列: 原方法ACC, C列: 改进方法ACC
    'NMI': {'orig_col': 3, 'imp_col': 4},  # D列: 原方法NMI, E列: 改进方法NMI
    'ARI': {'orig_col': 5, 'imp_col': 6}  # F列: 原方法ARI, G列: 改进方法ARI
}


def extract_all_data():
    """提取所有数据集的原方法和改进方法的指标值"""
    # 读取Excel文件
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

    # 存储所有数据
    all_data = {}

    # 对于每种方法
    for n, method_info in methods.items():
        method_data = {}

        # 对于每个指标
        for metric, cols in metrics.items():
            metric_data = []

            # 对于每个数据集 (1-16)
            for i in range(1, 17):
                # 计算行索引 (Excel行号从1开始，pandas索引从0开始)
                row_idx = 2 + i + 22 * (n - 1)  # 第3行对应索引2

                # 提取原方法和改进方法的指标值
                orig_val = df.iloc[row_idx, cols['orig_col']]  # 原方法列
                imp_val = df.iloc[row_idx, cols['imp_col']]  # 改进方法列

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


def perform_significance_test(all_data):
    """对每个方法和每个指标进行配对样本t检验"""
    results = {}

    print("\n显著性分析结果 (配对样本t检验):")
    print("=" * 70)

    for n, method_info in methods.items():
        method_name = method_info['name']
        method_data = all_data[n]

        results[method_name] = {}

        print(f"\n方法: {method_name}")
        print("-" * 60)

        for metric, data_list in method_data.items():
            # 提取原方法和改进方法的数值
            orig_vals = [d['orig_val'] for d in data_list]
            imp_vals = [d['imp_val'] for d in data_list]

            # 计算差值
            differences = [imp_val - orig_val for imp_val, orig_val in zip(imp_vals, orig_vals)]

            # 执行配对样本t检验 (单尾检验，备择假设：改进方法 > 原方法)
            t_stat, p_value = stats.ttest_rel(imp_vals, orig_vals, alternative='greater')

            # 计算效果量 (Cohen's d)
            mean_diff = np.mean(differences)
            std_diff = np.std(differences, ddof=1)  # 样本标准差
            cohens_d = mean_diff / std_diff if std_diff != 0 else 0

            # 存储结果
            result = {
                't_stat': t_stat,
                'p_value': p_value,
                'mean_diff': mean_diff,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05,
                'num_datasets': len(data_list)
            }

            results[method_name][metric] = result

            # 输出结果
            significance = "显著提升" if result['significant'] else "未显著提升"
            print(f"指标: {metric}")
            print(f"  数据集数量: {result['num_datasets']}")
            print(f"  t值: {t_stat:.4f}, p值: {p_value:.6f}")
            print(f"  平均提升: {mean_diff:.4f}, Cohen's d: {cohens_d:.4f}")
            print(f"  结果: {significance}")
            print("-" * 40)

    print("=" * 70)
    return results


def plot_significance_results(results):
    """可视化显著性分析结果"""
    # 为每个指标创建图表
    for metric in metrics.keys():
        plt.figure(figsize=(12, 8))

        # 提取数据
        method_names = []
        t_stats = []
        p_values = []
        mean_diffs = []

        for method_name, metrics_data in results.items():
            if metric in metrics_data:
                method_names.append(method_name)
                t_stats.append(metrics_data[metric]['t_stat'])
                p_values.append(metrics_data[metric]['p_value'])
                mean_diffs.append(metrics_data[metric]['mean_diff'])

        # 创建双轴图表
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # 柱状图：平均提升
        bars = ax1.bar(method_names, mean_diffs, color='#1E90FF', alpha=0.7)
        ax1.set_xlabel('聚类方法')
        ax1.set_ylabel('平均提升', color='#1E90FF')
        ax1.tick_params(axis='y', labelcolor='#1E90FF')

        # 添加p值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            sign = '*' if p_values[i] < 0.05 else ''
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                     f"{mean_diffs[i]:.4f}{sign}\n(p={p_values[i]:.4f})",
                     ha='center', va='bottom', fontsize=9)

        # 创建第二个Y轴：t值
        ax2 = ax1.twinx()
        ax2.plot(method_names, t_stats, 'ro-', linewidth=2, markersize=8)
        ax2.set_ylabel('t值', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # 添加显著性水平线
        ax2.axhline(y=1.96, color='gray', linestyle='--', alpha=0.7)
        ax2.text(0.5, 2.0, 'α=0.05显著性水平', color='gray', fontsize=10)

        plt.title(f'聚类方法改进前后{metric}指标显著性分析', fontsize=14, pad=20)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()

        # 保存图像
        output_dir = '显著性分析'
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'{metric}_显著性分析.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存显著性分析图: {save_path}")

        plt.show()


# 主程序
if __name__ == "__main__":
    print(f"从文件 {excel_path} 的 {sheet_name} 中提取数据...")
    all_data = extract_all_data()

    # 进行显著性分析
    results = perform_significance_test(all_data)

    # 可视化显著性分析结果
    print("\n正在生成显著性分析图表...")
    plot_significance_results(results)

    print("\n显著性分析完成！")