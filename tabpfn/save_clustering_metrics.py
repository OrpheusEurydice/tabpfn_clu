import os
import matplotlib.pyplot as plt
from matplotlib import rcParams


def plot_nmi_trend(nmi_list, algorithm_name, dataset_number=1, save_dir=None):
    """
    绘制并保存NMI指标变化趋势图

    Parameters:
        nmi_list (list): NMI指标值列表
        algorithm_name (str): 使用的聚类算法名称
        dataset_number (int): 数据集编号（默认1）
        save_dir (str): 可选指定保存目录，默认使用预设路径

    Returns:
        str: 生成的图片完整路径

    Raises:
        ValueError: 当输入数据不符合要求时
    """
    # 参数校验
    if not isinstance(nmi_list, list) or len(nmi_list) == 0:
        raise ValueError("nmi_list必须为非空列表")
    if not all(isinstance(x, (float, int)) for x in nmi_list):
        raise ValueError("nmi_list包含非数值元素")

    # 设置默认存储路径
    if save_dir is None:
        #save_dir = fr"D:\所有数据每轮聚类结果(未归一化)\{dataset_number}_results"
        save_dir = fr"D:\性能超4个的数据集每轮结果(归一化)\{dataset_number}_results"
    os.makedirs(save_dir, exist_ok=True)

    # 配置可视化参数
    rcParams['font.sans-serif'] = ['SimHei']
    rcParams['axes.unicode_minus'] = False

    # 创建画布
    plt.figure(figsize=(10, 6), dpi=100)

    # 生成横坐标标签
    epochs = [f"{i}" for i in range(len(nmi_list))]

    # 绘制主折线
    main_line = plt.plot(epochs, nmi_list,
                         color='#1f77b4',
                         marker='s',
                         linestyle='-.',
                         linewidth=2,
                         markersize=8,
                         markerfacecolor='#ff7f0e',
                         markeredgewidth=1.5)

    # 添加数据标签
    for i, val in enumerate(nmi_list):
        plt.annotate(f"{val:.4f}",
                     xy=(i, val),
                     xytext=(0, 8),
                     textcoords='offset points',
                     ha='center',
                     fontsize=9,
                     color='#2ca02c')

    # 装饰图表
    plt.title(f"数据集 #{dataset_number} - {algorithm_name}\nNMI指标迭代趋势",
              fontsize=14, pad=20)
    plt.xlabel("训练轮次", fontsize=12, labelpad=10)
    plt.ylabel("NMI值", fontsize=12, labelpad=10)

    # 动态计算纵轴范围
    nmi_min = min(nmi_list)
    nmi_max = max(nmi_list)

    # 计算范围时考虑数据全为同一值的情况
    if nmi_min == nmi_max:
        y_lower = max(0.0, nmi_min - 0.1)  # 保证最小值不低于0
        y_upper = min(1.0, nmi_max + 0.1)  # 保证最大值不超过1
    else:
        # 常规情况：上下各留5%的余量
        data_range = nmi_max - nmi_min
        y_lower = max(0.0, nmi_min - data_range * 0.05)
        y_upper = min(1.0, nmi_max + data_range * 0.05)

    # 设置纵轴范围
    plt.ylim(y_lower, y_upper)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 保存文件
    filename = f"ds{dataset_number}_{algorithm_name}_nmi_trend.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    return save_path

def save_clustering_metrics(acc_list, nmi_list, ari_list, dataset_number, algorithm_name):
    """
    保存聚类指标结果，包含轮次标题和四位小数精度

    Parameters:
        acc_list (list): 准确率列表，元素为浮点数
        nmi_list (list): 标准化互信息列表，元素为浮点数
        ari_list (list): 调整兰德指数列表，元素为浮点数
        algorithm_name (str): 使用的聚类算法名称
        dataset_number (int): 数据集编号

    Returns:
        str: 生成的完整文件名

    Raises:
        ValueError: 当输入列表长度不一致时
    """
    # 校验列表长度
    list_len = len(acc_list)
    if not (len(nmi_list) == len(ari_list) == list_len):
        raise ValueError("所有指标列表必须具有相同长度")

    # 生成轮次标题行
    round_header = '\t'.join([f"Round{i}" for i in range(list_len)])

    # 格式化数值为四位小数
    format_value = lambda x: f"{x:.4f}"
    acc_formatted = list(map(format_value, acc_list))
    nmi_formatted = list(map(format_value, nmi_list))
    ari_formatted = list(map(format_value, ari_list))

    # 创建目标目录
    #base_dir = fr"D:\所有数据每轮聚类结果(未归一化)\{dataset_number}_results"
    base_dir = fr"D:\性能超4个的数据集每轮结果(归一化)\{dataset_number}_results"
    os.makedirs(base_dir, exist_ok=True)

    # 生成文件名
    filename = f"ds{dataset_number}_{algorithm_name}_metrics.txt"

    full_path = os.path.join(base_dir, filename)


    # 写入文件
    with open(full_path, 'w') as f:
        f.write(round_header + '\n')
        f.write('\t'.join(acc_formatted) + '\n')
        f.write('\t'.join(nmi_formatted) + '\n')
        f.write('\t'.join(ari_formatted) + '\n')

    # 调用绘图函数
    chart_path = plot_nmi_trend(nmi_list, algorithm_name, dataset_number)


    return base_dir