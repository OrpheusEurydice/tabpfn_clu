import numpy as np
import os
import pandas as pd
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


def custom_clustering(data, k, data_all=None, clustering_method = 'kmeans', max_epoch = 17, dataset_name = 'Flame', y=None):
    # 初始化时生成原始数据图
    #visualize_raw_data(data_all, dataset_name, y)

    # 生成参数范围（6个等间距值）
    params = np.linspace(0.4, 0.9, 6)

    # 创建结果容器
    #results = []

    # 储存当前方法每一轮的评价指标
    accuracies = []
    nmi = []
    ari = []

    best_nmi = 0
    best_predictions = None

    # 参数敏感性分析（两重循环）
    for pi1 in params:
        for pi2 in params:
            sampled_data, _ = train_test_split(data, test_size=1 - pi1, random_state=42)
            sampled_data_y = None

            # 用kmeans的结果作为标签训练模型
            if clustering_method == 'kmeans':
                # 使用 KMeans 聚类
                print("Using KMeans for clustering...")
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                # kmeans = KMeans(n_clusters=train_k, random_state=42, n_init='auto')
                sampled_data_y = kmeans.fit_predict(sampled_data)

            # 用谱聚类的结果作为标签训练模型
            elif clustering_method == 'spectral':
                # 使用 SpectralClustering 聚类
                print("Using SpectralClustering for clustering...")
                spectral = SpectralClustering(n_clusters=k, random_state=42)
                # spectral = SpectralClustering(n_clusters=train_k, random_state=42)
                sampled_data_y = spectral.fit_predict(sampled_data)

            # 用层次聚类的结果作为标签训练模型
            elif clustering_method == 'hierarchical':
                print("Using HierarchicalClustering for clustering...")
                hierarchical = AgglomerativeClustering(n_clusters=k)
                # hierarchical = AgglomerativeClustering(n_clusters=train_k)
                sampled_data_y = hierarchical.fit_predict(sampled_data)


            # 用高斯混合模型聚类的结果作为标签训练模型
            elif clustering_method == 'gmm':
                print("Using GMM for clustering...")
                gmm = GaussianMixture(n_components=k, random_state=42)
                # gmm = GaussianMixture(n_components=train_k, random_state=42)
                sampled_data_y = gmm.fit_predict(sampled_data)

            # 用AP聚类的结果作为标签训练模型
            elif clustering_method == 'ap':
                print("Using AP for clustering...")
                sampled_data_y = apgpt(sampled_data, k)


            elif clustering_method == 'dp':
                print("Using DP for clustering...")
                sampled_data_y = cluster_dp_k(sampled_data, k)


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

            clf = TabPFNClassifier(device='cuda', random_state=42)

            flag = 1

            """用聚类结果作为训练集标签,训练tabpfn模型"""
            clf.fit(sampled_data, sampled_data_y)

            # predictions = batch_predict(clf, data)

            predictions = batch_predict(clf, data_all)
            # predictions_first2w = predictions[:20000]


            while True:
                print(f"Epoch {flag}:")

                flag += 1
                previous_labels = predictions.copy()
                # previous_first2w_labels = predictions_first2w.copy()

                """把上一轮的预测结果作为标签再划分出新一轮的训练集，用来训练新一轮的tabpfn参数"""
                X_train, _, y_train, _ = train_test_split(data, previous_labels, test_size=1 - pi2, random_state=flag)
                # X_train, _, y_train, _ = train_test_split(data, previous_first2w_labels, test_size=0.6, random_state=flag)

                clf.fit(X_train, y_train)

                # predictions = batch_predict(clf, data)

                predictions = batch_predict(clf, data_all)
                # predictions_first2w = predictions[:20000]

                # 可视化当前轮次结果
                # visualize_clustering(flag, data_all, predictions, dataset_name)

                # 计算 Pseudo accuracy
                acc = cluster_accuracy(previous_labels, predictions)
                # pseudo_accuracies.append(acc)
                print("Pseudo accuracy:\n", acc)


                # 如果聚类结果一致，则停止迭代
                if np.array_equal(previous_labels, predictions) or flag > max_epoch:
                    break

            acc_true = cluster_accuracy(y, predictions)
            nmi_true = normalized_mutual_info_score(y, predictions)
            ari_true = adjusted_rand_score(y, predictions)
            accuracies.append(acc_true)
            nmi.append(nmi_true)
            ari.append(ari_true)

            if nmi_true > best_nmi:
                best_nmi = nmi_true
                best_predictions = predictions




    return best_predictions, accuracies, nmi, ari
