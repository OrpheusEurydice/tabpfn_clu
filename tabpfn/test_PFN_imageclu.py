import time

import numpy as np
# from clustering_PFN import custom_clustering
from PFN2_2 import custom_clustering
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, AffinityPropagation
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from tabpfn import TabPFNClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import RobustScaler

from gmm_accuracy import gmm_accuracy
from save_clustering_metrics import save_clustering_metrics


def cluster_accuracy(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)
    return np.max(cm, axis=0).sum() / np.sum(cm)

def process_data():
    #file_number = 16

    folder = "D:/tabpfn/output"
    file_path = f"{folder}/combined_dataset_100d_VGG.csv"

    print(f"\nProcessing file: {file_path}")

    # 加载数据
    df = pd.read_csv(file_path, delimiter=",")

    y = df.iloc[:, -1].values
    y = np.array(y)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    k = len(np.unique(y))


    # 全体特征
    X = df.iloc[:, :-1].values

    # robust_scaler = RobustScaler(quantile_range=(25, 75))  # 使用四分位数间距
    # X = robust_scaler.fit_transform(X)

    # 创建Min-Max归一化器（默认缩放到[0,1]）
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))  # 可省略参数，默认即为(0,1)
    X = minmax_scaler.fit_transform(X)

    # 前一万行的特征
    X_train = X[:10000]

    # 初始化计时字典
    time_records = {}

    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)



    #自定义聚类算法
    # # 1.用kmeans聚类标签训练的结果
    # # 数据集规模大于10000时
    # pred_custom = custom_clustering(X_train, k, X, clustering_method='kmeans')
    # print(f"Using kmeans accuracy:{cluster_accuracy(y, pred_custom)}")
    # print(f"Using kmeans NMI:{normalized_mutual_info_score(y, pred_custom)}")
    # print(f"Using kmeans ARI:{adjusted_rand_score(y, pred_custom)}")
    #
    # # 2.用层次聚类标签训练的结果
    # # 数据集规模大于10000时
    # pred_custom_hierarchical = custom_clustering(X_train, k, X, clustering_method='hierarchical')
    # print(f"Using agglomerative accuracy:{cluster_accuracy(y, pred_custom_hierarchical)}")
    # print(f"Using agglomerative NMI:{normalized_mutual_info_score(y, pred_custom_hierarchical)}")
    # print(f"Using agglomerative ARI:{adjusted_rand_score(y, pred_custom_hierarchical)}")
    #
    # # 3.用谱聚类标签训练的结果
    # # 数据集规模大于10000时
    # pred_custom_spectral = custom_clustering(X_train, k, X, clustering_method='spectral')
    # print(f"Using spectral accuracy:{cluster_accuracy(y, pred_custom_spectral)}")
    # print(f"Using spectral NMI:{normalized_mutual_info_score(y, pred_custom_spectral)}")
    # print(f"Using spectral ARI:{adjusted_rand_score(y, pred_custom_spectral)}")

    # 4.用GMM聚类标签训练的结果
    # 数据集规模大于10000时
    # 数据集规模大于10000时
    start_time = time.time()
    pred_custom_gmm, acc_list4, nmi_list4, ari_list4 = custom_clustering(X_train, k, X, clustering_method='gmm', y=y)
    time_records['custom_gmm'] = time.time() - start_time
    print("Time of custom_gmm:", time_records['custom_gmm'])
    print(f"Using gmm accuracy:{cluster_accuracy(y, pred_custom_gmm)}")
    # 调用函数
    file_path = save_clustering_metrics(
        acc_list=acc_list4,
        nmi_list=nmi_list4,
        ari_list=ari_list4,
        algorithm_name="gmm",
        dataset_number=1
    )
    print(f"gmm结果已保存到: {file_path}")

    # # 5.用AP聚类标签训练的结果
    # pred_custom_ap = custom_clustering(X_train, k, X, clustering_method='ap')
    # print(f"Using ap accuracy:{cluster_accuracy(y, pred_custom_ap)}")
    # print(f"Using ap NMI:{normalized_mutual_info_score(y, pred_custom_ap)}")
    # print(f"Using ap ARI:{adjusted_rand_score(y, pred_custom_ap)}")
    #
    # # 6.用Density-Peak Clustering (DP)聚类标签训练的结果
    # pred_custom_dp = custom_clustering(X_train, k, X, clustering_method='dp')
    # print(f"Using dp accuracy:{cluster_accuracy(y, pred_custom_dp)}")
    # print(f"Using dp NMI:{normalized_mutual_info_score(y, pred_custom_dp)}")
    # print(f"Using dp ARI:{adjusted_rand_score(y, pred_custom_dp)}")
    #
    # # 7.用Fuzzy C-means聚类标签训练的结果
    # pred_custom_fcm = custom_clustering(X_train, k, X, clustering_method='fcm')
    # print(f"Using fcm accuracy:{cluster_accuracy(y, pred_custom_fcm)}")
    # print(f"Using fcm NMI:{normalized_mutual_info_score(y, pred_custom_fcm)}")
    # print(f"Using fcm ARI:{adjusted_rand_score(y, pred_custom_fcm)}")
    #
    # # 8.用Border-Peeling Clustering (BPC)聚类标签训练的结果
    # pred_custom_bpc = custom_clustering(X_train, k, X, clustering_method='bpc')
    # print(f"Using bpc accuracy:{cluster_accuracy(y, pred_custom_bpc)}")
    # print(f"Using bpc NMI:{normalized_mutual_info_score(y, pred_custom_bpc)}")
    # print(f"Using bpc ARI:{adjusted_rand_score(y, pred_custom_bpc)}")


    # 对比实验
    num_runs = 10
    # 1. K-means 聚类
    # acc_kmeans, nmi_kmeans, ari_kmeans = kmeans_accuracy(X, y, k, num_runs)
    #
    # # 2. 层次聚类
    # print("Running AgglomerativeClustering...")
    # agg = AgglomerativeClustering(n_clusters=k)
    # pred_hierarchical = agg.fit_predict(X)
    # acc_hierarchical = cluster_accuracy(y, pred_hierarchical)
    # nmi_hierarchical = normalized_mutual_info_score(y, pred_hierarchical)
    # ari_hierarchical = adjusted_rand_score(y, pred_hierarchical)
    # print(f"Accuracy: {acc_hierarchical:.4f}")
    #
    # # 3. 谱聚类
    # #acc_spectral, nmi_spectral, ari_spectral = spectral_accuracy(X, y, k, num_runs)
    # acc_spectral, nmi_spectral, ari_spectral = spectral_accuracy(X, y, k)
    #
    # # 4. 基于密度的聚类
    # #acc_dbscan, nmi_dbscan, ari_dbscan = dbscan_accuracy(X, y, k)
    #
    # 5. 高斯混合模型聚类
    acc_gmm, nmi_gmm, ari_gmm, time_gmm = gmm_accuracy(X, y, k, 2)
    print("Time of gmm:", time_gmm)
    #acc_gmm, nmi_gmm, ari_gmm, time_gmm = gmm_accuracy(X, y, k, num_runs)

    # 6. AP聚类
    #print("Running APClustering...")
    # ap = AffinityPropagation(random_state=42)
    # micro_clusters = ap.fit_predict(X)
    #
    # # 层次聚类合并到 k 类
    # agg = AgglomerativeClustering(n_clusters=k)
    # # 忽略噪声
    # pred_ap = agg.fit_predict(X[micro_clusters != -1])

    #pred_ap = apgpt(X, k)

    # acc_ap = cluster_accuracy(y, pred_ap)
    # nmi_ap = normalized_mutual_info_score(y, pred_ap)
    # ari_ap = adjusted_rand_score(y, pred_ap)
    # print(f"Accuracy: {acc_ap:.4f}")

    # 7.gk-means聚类
    # acc_gkmeans, nmi_gkmeans, ari_gkmeans = gkmeans_accuracy(X, y, k, num_runs)

    # # 8.Fuzzy C-means聚类
    # acc_fcm, nmi_fcm, ari_fcm = fcm_accuracy(X, y, k, num_runs)

    # # 9.Border-Peeling Clustering (BPC)聚类
    # print("Running Border-Peeling Clustering...")
    # #pred_bpc = border_peeling_clustering(X, k=10, alpha=0.5, min_cluster_size=10, max_iter=50)
    # # 运行参数搜索（目标k个簇）
    # pred_bpc = optimized_bpc_search(
    #     X,
    #     target_clusters=k,  # 期望得到2个月牙形簇
    #     alpha_range=(0.1, 0.5),
    #     k_range=(10, 20),
    #     verbose=True
    # )
    # acc_bpc = cluster_accuracy(y, pred_bpc)
    # nmi_bpc = normalized_mutual_info_score(y, pred_bpc)
    # ari_bpc = adjusted_rand_score(y, pred_bpc)
    # print(f"Accuracy: {acc_bpc:.4f}")

    # # 10.Density-Peak 聚类（dp聚类）
    # print("Running Density-Peak Clustering...")
    # pred_dp = cluster_dp_k(X, k)
    # acc_dp = cluster_accuracy(y, pred_dp)
    # nmi_dp = normalized_mutual_info_score(y, pred_dp)
    # ari_dp = adjusted_rand_score(y, pred_dp)
    # print(f"Accuracy: {acc_dp:.4f}")


    # 计算评价指标
    # models = ['Custom_kmeans', 'K-means', 'Custom_hierarchical', 'Agglomerative',
    #           'Custom_spectral', 'Spectral', 'Custom_gmm', 'GMM']
    models = [
              #'Custom_kmeans', 'Custom_hierarchical',
                'Custom_gmm', 'gmm',
                #'Custom_ap', 'Custom_dp',
                #'Custom_fcm', 'Custom_bpc'
                ]
    accuracies = [
        # cluster_accuracy(y, pred_custom),
        # cluster_accuracy(y, pred_custom_hierarchical),
        # cluster_accuracy(y, pred_custom_spectral),
        cluster_accuracy(y, pred_custom_gmm),
        acc_gmm
        # cluster_accuracy(y, pred_custom_ap),
        # cluster_accuracy(y, pred_custom_dp),
        # cluster_accuracy(y, pred_custom_fcm),
        # cluster_accuracy(y, pred_custom_bpc)
    ]
    nmi_scores = [
        # normalized_mutual_info_score(y, pred_custom),
        # normalized_mutual_info_score(y, pred_custom_hierarchical),
        # normalized_mutual_info_score(y, pred_custom_spectral),
        normalized_mutual_info_score(y, pred_custom_gmm),
        nmi_gmm

    ]
    ari_scores = [
        # adjusted_rand_score(y, pred_custom),
        # adjusted_rand_score(y, pred_custom_hierarchical),
        # adjusted_rand_score(y, pred_custom_spectral),
        adjusted_rand_score(y, pred_custom_gmm),
        ari_gmm
        # adjusted_rand_score(y, pred_custom_ap),
        # adjusted_rand_score(y, pred_custom_dp),
        # adjusted_rand_score(y, pred_custom_fcm),
        # adjusted_rand_score(y, pred_custom_bpc),
    ]

    # 打印评价指标对比表格
    df_results = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracies,
        'NMI': nmi_scores,
        'ARI': ari_scores
    })

    print("\n聚类评价指标对比表格：")
    print(df_results.round(4))

    #write_transposed_results_to_excel(df_results, data_set_index=file_number)


if __name__ == "__main__":
    print("Clustering imagedatas from cifar-10 after reducing dimensions to 100 by VGG:")
    process_data()