import matplotlib
import numpy as np
import os
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN, AffinityPropagation
from sklearn.datasets import load_breast_cancer, load_iris, make_blobs, make_moons, make_circles
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier
import matplotlib.pyplot as plt

from ap_gpt import apgpt
from bpc_implementation import border_peeling_clustering, optimized_bpc_search
from dp_implementation import cluster_dp_k

# FCM (Fuzzy C-Means) 聚类
import skfuzzy as fuzz  # 需要安装: pip install scikit-fuzzy

from sklearn.feature_selection import VarianceThreshold


def cluster_accuracy(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)
    return np.max(cm, axis=0).sum() / np.sum(cm)

def batch_predict(model, data, batch_size=500):
    predictions = []
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        batch_pred = model.predict(batch)  # 逐批预测
        predictions.extend(batch_pred)
    return np.array(predictions)

# ------------------------- 可视化函数 -------------------------
def visualize_raw_data1(data, name, y, train_idx=None):
    """生成白色背景的原始数据分布图，支持高亮训练集"""
    output_dir = f"D:/流程图实例/{name.upper()}_Results"
    os.makedirs(output_dir, exist_ok=True)

    # 降维处理
    if data.shape[1] > 2:
        pca = PCA(n_components=2)
        vis_data = pca.fit_transform(data)
    else:
        vis_data = data

    # 创建画布并设置背景
    plt.figure(figsize=(8, 6))
    # ax = plt.gca()
    # ax.set_facecolor('white')

    #色调1
    # 高对比度配色
    custom_cmap = plt.cm.get_cmap('RdYlBu', 2)

    # 色调2
    # 特殊样式处理
    num_classes = len(np.unique(y))
    cmap = plt.cm.get_cmap('tab20', 20)  # 固定为20色
    colors = cmap(np.arange(20))  # 提取全部颜色
    # 新颜色代码（例如替换为红色）
    new_color = "#FF0000"
    new_rgba = matplotlib.colors.to_rgba(new_color)
    colors[6] = new_rgba  # 替换索引6
    mcmap = ListedColormap(colors[:num_classes])  # 按实际类别数截取

    # 根据是否传入训练集索引进行不同绘制逻辑
    if train_idx is not None:
        # 创建布尔掩码
        mask = np.zeros(len(data), dtype=bool)
        mask[train_idx] = True

        # 先绘制非训练集（浅灰色）
        non_train_data = vis_data[~mask]
        plt.scatter(
            non_train_data[:, 0], non_train_data[:, 1],
            c='lightgrey', s=15, alpha=0.8,
            edgecolor='white', linewidth=0.3
        )

        # 再绘制训练集（按标签着色）
        train_data = vis_data[mask]
        if y is not None:
            scatter = plt.scatter(
                train_data[:, 0], train_data[:, 1],
                c=y,
                # c='black',
                cmap=mcmap,
                #cmap=custom_cmap,
                s=15, alpha=0.8,
                edgecolor='white', linewidth=0.3
            )
        else:
            scatter = plt.scatter(
                train_data[:, 0], train_data[:, 1],
                c='black',
                cmap=custom_cmap, s=15, alpha=0.8,
                edgecolor='white', linewidth=0.3
            )

    else:
        # 原始绘制方式（全部按标签着色）
        scatter = plt.scatter(
            vis_data[:, 0], vis_data[:, 1],
            c='lightgrey', cmap=custom_cmap, s=15, alpha=0.8,
            edgecolor='white', linewidth=0.3
        )

    # 颜色条设置
    cbar = plt.colorbar(scatter, boundaries=np.arange(len(np.unique(y)) + 1) - 0.5)
    cbar.set_ticks(np.arange(len(np.unique(y))))


    if train_idx is not None:
        #plt.title(f"Raw Data Train Distribution\nDimensions: {data.shape[1]}", fontsize=10)
        plt.savefig(f"{output_dir}/RAW_DATA_train.png", dpi=600, bbox_inches='tight', transparent=False)
        plt.savefig(f"{output_dir}/RAW_DATA_train.eps", format='eps', dpi=600, bbox_inches='tight')
        plt.close()
        print(f"原始数据图已保存至: {output_dir}/RAW_DATA_train.png")
    else:
        #plt.title(f"Raw Data Distribution\nDimensions: {data.shape[1]}", fontsize=10)
        plt.savefig(f"{output_dir}/RAW_DATA.png", dpi=600, bbox_inches='tight', transparent=False)
        plt.savefig(f"{output_dir}/RAW_DATA.eps", format='eps', dpi=600, bbox_inches='tight')
        plt.close()
        print(f"原始数据图已保存至: {output_dir}/RAW_DATA.png")
    if y is not None:
        plt.savefig(f"{output_dir}/RAW_DATA_train_result.png", dpi=600, bbox_inches='tight')
        plt.savefig(f"{output_dir}/RAW_DATA_train_result.eps", format='eps', dpi=600, bbox_inches='tight')
        plt.close()
        print(f"原始数据图已保存至: {output_dir}/RAW_DATA_train_result.png")

def visualize_train_data1(data, name, train_y, train_idx=None):
    """生成白色背景的原始数据分布图，支持高亮训练集"""
    output_dir = f"D:/流程图实例/{name.upper()}_Results"
    os.makedirs(output_dir, exist_ok=True)

    # 降维处理
    if data.shape[1] > 2:
        pca = PCA(n_components=2)
        vis_data = pca.fit_transform(data)
    else:
        vis_data = data

    # 创建画布并设置背景
    plt.figure(figsize=(8, 6))
    # ax = plt.gca()
    # ax.set_facecolor('white')

    # 特殊样式处理
    num_classes = len(np.unique(train_y))
    cmap = plt.cm.get_cmap('tab20', 20)  # 固定为20色
    colors = cmap(np.arange(20))  # 提取全部颜色
    # 新颜色代码（例如替换为红色）
    new_color = "#FF0000"
    new_rgba = matplotlib.colors.to_rgba(new_color)
    colors[6] = new_rgba  # 替换索引6
    mcmap = ListedColormap(colors[:num_classes])  # 按实际类别数截取


    # 根据是否传入训练集索引进行不同绘制逻辑
    if train_idx is not None:
        # 创建布尔掩码
        mask = np.zeros(len(data), dtype=bool)
        mask[train_idx] = True

        # 先绘制非训练集（浅灰色）
        non_train_data = vis_data[~mask]
        plt.scatter(
            non_train_data[:, 0], non_train_data[:, 1],
            c='lightgrey', s=15, alpha=0.8,
            edgecolor='white', linewidth=0.3
        )

        # 再绘制训练集（按标签着色）
        train_data = vis_data[mask]
        if train_y is not None:
            scatter = plt.scatter(
                train_data[:, 0], train_data[:, 1],
                c=train_y,
                # c='black',
                cmap=mcmap, s=15, alpha=0.8,
                edgecolor='white', linewidth=0.3
            )

    # # 设置坐标轴边框颜色
    # for spine in ax.spines.values():
    #     spine.set_color('k')

    # 颜色条设置
    cbar = plt.colorbar(scatter,
                        boundaries=np.arange(len(np.unique(train_y)) + 1) - 0.5)
    cbar.set_ticks(np.arange(len(np.unique(train_y))))

    if train_y is not None:
        plt.savefig(f"{output_dir}/RAW_DATA_train_result.png", dpi=600, bbox_inches='tight')
        plt.savefig(f"{output_dir}/RAW_DATA_train_result.eps", format='eps', dpi=600, bbox_inches='tight')
        plt.close()
        print(f"原始数据图已保存至: {output_dir}/RAW_DATA_train_result.png")

def visualize_train_data2(data, name, train_y, train_idx=None):
    """生成白色背景的原始数据分布图，支持高亮训练集"""
    output_dir = f"D:/流程图实例/{name.upper()}_Results"
    os.makedirs(output_dir, exist_ok=True)

    # 降维处理
    if data.shape[1] > 2:
        pca = PCA(n_components=2)
        vis_data = pca.fit_transform(data)
    else:
        vis_data = data

    # 创建画布并设置背景
    plt.figure(figsize=(8, 6))
    # ax = plt.gca()
    # ax.set_facecolor('white')

    # 特殊样式处理
    num_classes = len(np.unique(train_y))
    cmap = plt.cm.get_cmap('tab20', 20)  # 固定为20色
    colors = cmap(np.arange(20))  # 提取全部颜色
    # 新颜色代码（例如替换为红色）
    new_color = "#FF0000"
    new_rgba = matplotlib.colors.to_rgba(new_color)
    colors[6] = new_rgba  # 替换索引6
    mcmap = ListedColormap(colors[:num_classes])  # 按实际类别数截取


    # 根据是否传入训练集索引进行不同绘制逻辑
    if train_idx is not None:
        # 创建布尔掩码
        mask = np.zeros(len(data), dtype=bool)
        mask[train_idx] = True

        # 先绘制非训练集（浅灰色）
        non_train_data = vis_data[~mask]
        plt.scatter(
            non_train_data[:, 0], non_train_data[:, 1],
            c='lightgrey', s=15, alpha=0.8,
            edgecolor='white', linewidth=0.3
        )

        # 再绘制训练集（按标签着色）
        train_data = vis_data[mask]
        if train_y is not None:
            scatter = plt.scatter(
                train_data[:, 0], train_data[:, 1],
                c=train_y,
                # c='black',
                cmap=mcmap, s=15, alpha=0.8,
                edgecolor='white', linewidth=0.3
            )

    # 颜色条设置
    cbar = plt.colorbar(scatter,
                        boundaries=np.arange(len(np.unique(train_y)) + 1) - 0.5)
    cbar.set_ticks(np.arange(len(np.unique(train_y))))

    if train_y is not None:
        plt.savefig(f"{output_dir}/RAW_DATA_train_result2.png", dpi=600, bbox_inches='tight')
        plt.savefig(f"{output_dir}/RAW_DATA_train_result2.eps", format='eps', dpi=600, bbox_inches='tight')
        plt.close()
        print(f"原始数据图已保存至: {output_dir}/RAW_DATA_train_result2.png")

def visualize_clustering(data, name, y, epoch):
    """生成白色背景的原始数据分布图，支持高亮训练集"""
    output_dir = f"D:/流程图实例/{name.upper()}_Results"
    os.makedirs(output_dir, exist_ok=True)

    # 创建画布并设置背景
    plt.figure(figsize=(8, 6))
    # ax = plt.gca()
    # ax.set_facecolor('white')

    # 特殊样式处理
    num_classes = len(np.unique(y))
    cmap = plt.cm.get_cmap('tab20', 20)  # 固定为20色
    colors = cmap(np.arange(20))  # 提取全部颜色
    # 新颜色代码（例如替换为红色）
    new_color = "#FF0000"
    new_rgba = matplotlib.colors.to_rgba(new_color)
    colors[6] = new_rgba  # 替换索引6
    mcmap = ListedColormap(colors[:num_classes])  # 按实际类别数截取

    # 绘制全集（按标签着色）
    scatter = plt.scatter(
        data[:, 0], data[:, 1],
        c=y,
        # c='black',
        cmap=mcmap, s=15, alpha=0.8,
        edgecolor='white', linewidth=0.3
    )


    # 颜色条设置
    cbar = plt.colorbar(scatter, boundaries=np.arange(len(np.unique(y)) + 1) - 0.5)
    cbar.set_ticks(np.arange(len(np.unique(y))))


    plt.savefig(f"{output_dir}/{name}_epoch_{epoch:03d}.png", dpi=600, bbox_inches='tight')  # 关闭透明
    plt.savefig(f"{output_dir}/{name}_epoch_{epoch:03d}.eps", format='eps', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"第{epoch}轮结果已保存至: {output_dir}/{name}_epoch_{epoch:03d}.png")

def custom_clustering(data, k, data_all=None, clustering_method = 'kmeans', max_epoch = 30, dataset_name = 'Flame', y=None):
    # 初始化时生成原始数据图
    # visualize_raw_data1(data=data, name=dataset_name, y=None, train_idx=None)

    # 生成与数据行数相同的数字索引
    data_indices = np.arange(data.shape[0])  # 假设data是二维数组或DataFrame

    # 直接划分索引
    train_indices, _ = train_test_split(
        data_indices,
        test_size=0.5,
        random_state=42
    )
    # 对训练集索引进行排序 (关键修改点)
    train_indices = np.sort(train_indices)  # numpy版本
    # 通过索引获取对应的训练数据
    sampled_data = data[train_indices]  # NumPy数组的索引方式

    sampled_data, _ = train_test_split(data, test_size=0.5, random_state=42)


    # 调用可视化函数
    # visualize_raw_data1(data=data, name=dataset_name, y=None, train_idx=train_indices)
    visualize_raw_data1(data=data, name=dataset_name, y=y, train_idx=data_indices)

    sampled_data_y = None


    #用kmeans的结果作为标签训练模型
    if clustering_method == 'kmeans':
        # 使用 KMeans 聚类
        print("Using KMeans for clustering...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        #kmeans = KMeans(n_clusters=train_k, random_state=42, n_init='auto')
        sampled_data_y = kmeans.fit_predict(sampled_data)

    # 用谱聚类的结果作为标签训练模型
    elif clustering_method == 'spectral':
        # 使用 SpectralClustering 聚类
        print("Using SpectralClustering for clustering...")
        spectral = SpectralClustering(n_clusters=k, random_state=42)
        #spectral = SpectralClustering(n_clusters=train_k, random_state=42)
        sampled_data_y = spectral.fit_predict(sampled_data)

    # 用层次聚类的结果作为标签训练模型
    elif clustering_method == 'hierarchical':
        print("Using HierarchicalClustering for clustering...")
        hierarchical = AgglomerativeClustering(n_clusters=k)
        #hierarchical = AgglomerativeClustering(n_clusters=train_k)
        sampled_data_y = hierarchical.fit_predict(sampled_data)


    # 用高斯混合模型聚类的结果作为标签训练模型
    elif clustering_method == 'gmm':
        print("Using GMM for clustering...")
        gmm = GaussianMixture(n_components=k, random_state=42)
        #gmm = GaussianMixture(n_components=train_k, random_state=42)
        sampled_data_y = gmm.fit_predict(sampled_data)

    # 用AP聚类的结果作为标签训练模型
    elif clustering_method == 'ap':
        print("Using AP for clustering...")
        sampled_data_y = apgpt(sampled_data, k)


    elif clustering_method == 'dp':
        print("Using DP for clustering...")
        sampled_data_y = cluster_dp_k(sampled_data, k)

    elif clustering_method == 'bpc':
        print("Using BPC for clustering...")
        # 参数说明：
        # - k: 最近邻数量
        # - alpha: 边界剥离比例(0-1)
        # - min_cluster_size: 最小簇大小
        #sampled_data_y = border_peeling_clustering(sampled_data, k=10, alpha=0.5, min_cluster_size=10, max_iter=50)
        # 运行参数搜索（目标k个簇）
        sampled_data_y = optimized_bpc_search(
            sampled_data,
            target_clusters=k,  # 期望得到2个月牙形簇
            alpha_range=(0.1, 0.9),
            k_range=(10, 20),
            verbose=True
        )

    elif clustering_method == 'fcm':
        print("Using Fuzzy C-means for clustering...")
        # 参数说明：
        # - c: 簇数量
        # - m: 模糊系数(>1)，值越大越模糊
        # - error: 终止误差
        # - maxiter: 最大迭代次数
        cntr, u, *_ = fuzz.cluster.cmeans(
            sampled_data.T, c=k, m=2, error=1e-5, maxiter=1000, seed=42
        )
        sampled_data_y = np.argmax(u, axis=0)  # 转换为硬聚类标签

    #train_indices_sort = np.sort(train_indices)
    visualize_train_data1(data=data, name=dataset_name, train_y=sampled_data_y, train_idx=train_indices)

    clf = TabPFNClassifier(device='cuda',  random_state = 42)

    flag = 1

    """用聚类结果作为训练集标签,训练tabpfn模型"""
    clf.fit(sampled_data, sampled_data_y)

    #predictions = batch_predict(clf, data)

    predictions = batch_predict(clf, data_all)
    #predictions_first2w = predictions[:10000]

    # 可视化当前轮次结果
    #visualize_clustering(data=data, name=dataset_name, y=predictions, epoch=flag)

    # 初始化记录 Pseudo accuracy 的列表
    #pseudo_accuracies = []


    while True:
        print(f"Epoch {flag}:")

        flag += 1
        previous_labels = predictions.copy()
        #previous_first2w_labels = predictions_first2w.copy()


        # 生成与数据行数相同的数字索引
        data_indices = np.arange(data.shape[0])  # 假设data是二维数组或DataFrame

        # 直接划分索引
        train_indices, _ = train_test_split(
            data_indices,
            test_size=0.6,
            random_state=flag
        )
        # 对训练集索引进行排序 (关键修改点)
        train_indices_sort = np.sort(train_indices)  # numpy版本
        # 通过索引获取对应的训练数据
        X_train = data[train_indices]  # NumPy数组的索引方式
        y_train = previous_labels[train_indices]
        y_train_sort = previous_labels[train_indices_sort]

        if flag==2:
            visualize_train_data2(data=data, name=dataset_name, train_y=y_train_sort, train_idx=train_indices_sort)

        """把上一轮的预测结果作为标签再划分出新一轮的训练集，用来训练新一轮的tabpfn参数"""
        X_train, _, y_train, _ = train_test_split(data, previous_labels, test_size=0.6, random_state = flag)

        clf.fit(X_train, y_train)

        #predictions = batch_predict(clf, data)

        predictions = batch_predict(clf, data_all)
        #predictions_first2w = predictions[:10000]


        # 可视化当前轮次结果
        visualize_clustering(data=data, name=dataset_name, y=predictions, epoch=flag)

        # 计算 Pseudo accuracy
        acc = cluster_accuracy(previous_labels, predictions)
        #pseudo_accuracies.append(acc)
        print("Pseudo accuracy:\n", acc)


        # 如果聚类结果一致，则停止迭代
        if np.array_equal(previous_labels, predictions) or flag > max_epoch:
            break

    return predictions