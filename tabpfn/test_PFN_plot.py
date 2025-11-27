import os

import joblib
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tslearn.clustering import silhouette_score

# from clustering_PFN import custom_clustering
from PFN2_流程图实例 import custom_clustering
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, AffinityPropagation
from sklearn.datasets import load_breast_cancer, load_iris, make_blobs, make_moons, make_circles
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from tabpfn import TabPFNClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import RobustScaler



def cluster_accuracy(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)
    return np.max(cm, axis=0).sum() / np.sum(cm)


def process_data(folder_path=None):
    #file_number = 16
    folder = "D:/tabpfn/data"
    file_path = f"{folder}/2d2k.txt"
    print(f"\nProcessing file: {file_path}")

    base_name = "2d2k"
    #
    # 加载数据
    df = pd.read_csv(file_path, delimiter=r'\s+', header=None)

    # 全体特征
    X = df.iloc[:, 0:-1].values
    y = df.iloc[:, -1].values

    # y = df.iloc[:, -1].values
    y = np.array(y)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    k = len(np.unique(y))

    # 创建Min-Max归一化器（默认缩放到[0,1]）
    # minmax_scaler = MinMaxScaler(feature_range=(0, 1))  # 可省略参数，默认即为(0,1)
    # X = minmax_scaler.fit_transform(X)

    # 前一万行的特征
    X_train = X

    # 自定义聚类算法
    # 1.用kmeans聚类标签训练的结果
    # 数据集规模大于10000时
    pred_custom = custom_clustering(X_train, k, X, clustering_method='kmeans', dataset_name=base_name, y=y)
    # 评估聚类效果
    score = silhouette_score(X, pred_custom)
    print(f"{base_name}轮廓系数: {score:.3f}")
    print(f"Using kmeans accuracy:{cluster_accuracy(y, pred_custom)}")
    print(f"Using kmeans NMI:{normalized_mutual_info_score(y, pred_custom)}")
    print(f"Using kmeans ARI:{adjusted_rand_score(y, pred_custom)}")

    # for filename in os.listdir(folder_path):
    #     file_path = os.path.join(folder_path, filename)
    #     print(f"\nProcessing file: {file_path}")
    #
    #     base_name = os.path.splitext(filename)[0]
    #
    #
    #     # 加载数据
    #     df = pd.read_csv(file_path, delimiter=",")
    #
    #     if "feature_0" not in df.columns or "feature_1" not in df.columns:
    #         print(f"跳过 {base_name}：前两列不是 feature_0 和 feature_1")
    #         continue
    #
    #     X = df[["feature_0", "feature_1"]].values
    #     y = df["label"].values if "label" in df.columns else [0] * len(X)
    #
    #     #y = df.iloc[:, -1].values
    #     y = np.array(y)
    #     label_encoder = LabelEncoder()
    #     y = label_encoder.fit_transform(y)
    #
    #     k = len(np.unique(y))
    #
    #     # 创建Min-Max归一化器（默认缩放到[0,1]）
    #     minmax_scaler = MinMaxScaler(feature_range=(0, 1))  # 可省略参数，默认即为(0,1)
    #     X = minmax_scaler.fit_transform(X)
    #
    #     # 前一万行的特征
    #     X_train = X
    #
    #
    #     # 自定义聚类算法
    #     # 1.用kmeans聚类标签训练的结果
    #     # 数据集规模大于10000时
    #     pred_custom = custom_clustering(X_train, k, X, clustering_method='kmeans', dataset_name=base_name, y=y)
    #     # 评估聚类效果
    #     score = silhouette_score(X, pred_custom)
    #     print(f"{base_name}轮廓系数: {score:.3f}")
    #     print(f"Using kmeans accuracy:{cluster_accuracy(y, pred_custom)}")
    #     print(f"Using kmeans NMI:{normalized_mutual_info_score(y, pred_custom)}")
    #     print(f"Using kmeans ARI:{adjusted_rand_score(y, pred_custom)}")
    #     #
    #     #
    #     # # 2.用层次聚类标签训练的结果
    #     # # 数据集规模大于10000时
    #     # pred_custom_hierarchical = custom_clustering(X_train, k, X, clustering_method='hierarchical', dataset_name=name)
    #     # # 评估聚类效果
    #     # score = silhouette_score(X, pred_custom_hierarchical)
    #     # print(f"{name}轮廓系数: {score:.3f}")
    #     #
    #     # # 3.用谱聚类标签训练的结果
    #     # # 数据集规模大于10000时
    #     # pred_custom_spectral = custom_clustering(X_train, k, X, clustering_method='spectral', dataset_name=name)
    #     # # 评估聚类效果
    #     # score = silhouette_score(X, pred_custom_spectral)
    #     # print(f"{name}轮廓系数: {score:.3f}")
    #     #
    #     #
    #     # 4.用GMM聚类标签训练的结果
    #     # 数据集规模大于10000时
    #     # pred_custom_gmm = custom_clustering(X_train, k, X, clustering_method='gmm', dataset_name=base_name, y=y)
    #     # # 评估聚类效果
    #     # score = silhouette_score(X, pred_custom_gmm)
    #     # print(f"{base_name}轮廓系数: {score:.3f}")
    #     # print(f"Using gmm accuracy:{cluster_accuracy(y, pred_custom_gmm)}")
    #     # print(f"Using gmm NMI:{normalized_mutual_info_score(y, pred_custom_gmm)}")
    #     # print(f"Using gmm ARI:{adjusted_rand_score(y, pred_custom_gmm)}")
    #
    #     #
    #     #
    #     # # 5.用AP聚类标签训练的结果
    #     # pred_custom_ap = custom_clustering(X_train, k, X, clustering_method='ap', dataset_name=name)
    #     # # 评估聚类效果
    #     # score = silhouette_score(X, pred_custom_ap)
    #     # print(f"{name}轮廓系数: {score:.3f}")
    #     #
    #     # # 6.用Density-Peak Clustering (DP)聚类标签训练的结果
    #     # pred_custom_dp = custom_clustering(X_train, k, X, clustering_method='dp', dataset_name=name)
    #     # # 评估聚类效果
    #     # score = silhouette_score(X, pred_custom_dp)
    #     # print(f"{name}轮廓系数: {score:.3f}")
    #     #
    #     #
    #     # # 7.用Fuzzy C-means聚类标签训练的结果
    #     # pred_custom_fcm = custom_clustering(X_train, k, X, clustering_method='fcm', dataset_name=name)
    #     # # 评估聚类效果
    #     # score = silhouette_score(X, pred_custom_fcm)
    #     # print(f"{name}轮廓系数: {score:.3f}")
    #
    #     # # 8.用Border-Peeling Clustering (BPC)聚类标签训练的结果
    #     # pred_custom_bpc = custom_clustering(X_train, k, X, clustering_method='bpc', dataset_name=name)
    #     # # 评估聚类效果
    #     # score = silhouette_score(X, pred_custom_bpc)
    #     # print(f"{name}轮廓系数: {score:.3f}")
    #


if __name__ == "__main__":
    # print("Clustering 2d dataset and visualize:")
    # folder_path = "D:/tabpfn/synthetic_data/synthetic_datasets"
    # process_data(folder_path)
    process_data()