import time

import numpy as np
# from clustering_PFN import custom_clustering
from PFN2_2_finetune import custom_clustering
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, AffinityPropagation
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from tabpfn import TabPFNClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from ap_gpt import apgpt
from bpc_implementation import border_peeling_clustering, optimized_bpc_search
from fcm_accuracy import fcm_accuracy
from gmm_accuracy import gmm_accuracy
from kmeans_accuracy import kmeans_accuracy
from save_paraselect_metrics import save_clustering_metrics
from spectral_accuracy import spectral_accuracy
from write import write_transposed_results_to_excel
import gc

def cluster_accuracy(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)
    return np.max(cm, axis=0).sum() / np.sum(cm)

def process_data(file_number):
    #file_number = 16

    folder = "D:/data_type_data_all"
    file_path = f"{folder}/{file_number}.data"
    print(f"\nProcessing file: {file_path}")

    # 加载数据
    df = pd.read_csv(file_path, delimiter=r'\s+')

    #df_train = df
    #beyond_1w = False

    # 如果行数超过10000，只取前10000行
    if df.shape[0] > 10000:
        #df_train = df.iloc[:20000]
        print(f"Skipping file {file_number} as number of samples is greater than 10000.")
        return None
        #beyond_1w = True

    # if df.shape[0] < 35:
    #     print(f"Skipping file {file_number} as number of samples is smaller than 35.")
    #     return None

    y = df.iloc[:, -1].values
    y = np.array(y)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    k = len(np.unique(y))

    # 如果类别数k超过10，则跳过当前数据集
    if k > 10:
        print(f"Skipping file {file_number} as number of clusters (k) is greater than 10.")
        return None

    # 全体特征
    X = df.iloc[:, :-1].values

    # 前两万行的特征
    #X_train = df_train.iloc[:, :-1].values

    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # X = scaler.transform(X)

    # 创建Min-Max归一化器（默认缩放到[0,1]）
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))  # 可省略参数，默认即为(0,1)
    X = minmax_scaler.fit_transform(X)


    X_train = X

    # 初始化计时字典
    time_records = {}


    # 自定义聚类算法
    # 1.用kmeans聚类标签训练的结果
    # 数据集规模小于等于10000时
    # pred_custom = custom_clustering(X, k, clustering_method='kmeans')
    # 数据集规模大于10000时
    start_time = time.time()
    pred_custom, acc_list1, nmi_list1, ari_list1= custom_clustering(X_train, k, X, clustering_method='kmeans',y=y)
    time_records['custom_kmeans'] = time.time() - start_time
    print("Time of custom_kmeans:", time_records['custom_kmeans'])
    print(f"Using kmeans accuracy:{cluster_accuracy(y, pred_custom)}")
    # 调用函数
    file_path = save_clustering_metrics(
        acc_list=acc_list1,
        nmi_list=nmi_list1,
        ari_list=ari_list1,
        algorithm_name="kmeans",
        dataset_number=file_number
    )
    print(f"kmeans结果已保存到: {file_path}")


    # # 2.用层次聚类标签训练的结果
    # # 数据集规模小于等于10000时
    # #pred_custom_hierarchical = custom_clustering(X, k, clustering_method='hierarchical')
    # # 数据集规模大于10000时
    # start_time = time.time()
    # pred_custom_hierarchical, acc_list2, nmi_list2, ari_list2 = custom_clustering(X_train, k, X, clustering_method='hierarchical',y=y)
    # time_records['custom_hierarchical'] = time.time() - start_time
    # print("Time of custom_hierarchical:", time_records['custom_hierarchical'])
    # print(f"Using agglomerative accuracy:{cluster_accuracy(y, pred_custom_hierarchical)}")
    # # 调用函数
    # file_path = save_clustering_metrics(
    #     acc_list=acc_list2,
    #     nmi_list=nmi_list2,
    #     ari_list=ari_list2,
    #     algorithm_name="hierarchical",
    #     dataset_number=file_number
    # )
    # print(f"hierarchical结果已保存到: {file_path}")
    #
    # # 3.用谱聚类标签训练的结果
    # # 数据集规模小于等于10000时
    # #pred_custom_spectral = custom_clustering(X, k, clustering_method='spectral')
    # # 数据集规模大于10000时
    # start_time = time.time()
    # pred_custom_spectral, acc_list3, nmi_list3, ari_list3 = custom_clustering(X_train, k, X, clustering_method='spectral',y=y)
    # time_records['custom_spectral'] = time.time() - start_time
    # print("Time of custom_spectral:", time_records['custom_spectral'])
    # print(f"Using spectral accuracy:{cluster_accuracy(y, pred_custom_spectral)}")
    # # 调用函数
    # file_path = save_clustering_metrics(
    #     acc_list=acc_list3,
    #     nmi_list=nmi_list3,
    #     ari_list=ari_list3,
    #     algorithm_name="spectral",
    #     dataset_number=file_number
    # )
    # print(f"spectral结果已保存到: {file_path}")
    #
    #
    # # 4.用GMM聚类标签训练的结果
    # # 数据集规模小于等于10000时
    # #pred_custom_gmm = custom_clustering(X, k, clustering_method='gmm')
    # # 数据集规模大于10000时
    # start_time = time.time()
    # pred_custom_gmm, acc_list4, nmi_list4, ari_list4 = custom_clustering(X_train, k, X, clustering_method='gmm',y=y)
    # time_records['custom_gmm'] = time.time() - start_time
    # print("Time of custom_gmm:", time_records['custom_gmm'])
    # print(f"Using gmm accuracy:{cluster_accuracy(y, pred_custom_gmm)}")
    # # 调用函数
    # file_path = save_clustering_metrics(
    #     acc_list=acc_list4,
    #     nmi_list=nmi_list4,
    #     ari_list=ari_list4,
    #     algorithm_name="gmm",
    #     dataset_number=file_number
    # )
    # print(f"gmm结果已保存到: {file_path}")
    #
    # # 5.用AP聚类标签训练的结果
    # #pred_custom_ap = custom_clustering(X, k, clustering_method='ap')
    # # 记录custom_ap聚类时间
    # start_time = time.time()
    # pred_custom_ap, acc_list5, nmi_list5, ari_list5 = custom_clustering(X_train, k, X, clustering_method='ap',y=y)
    # time_records['custom_ap'] = time.time() - start_time
    # print("Time of custom_ap:", time_records['custom_ap'])  # 无f-string版本
    # print(f"Using ap accuracy:{cluster_accuracy(y, pred_custom_ap)}")
    # # 调用函数
    # file_path = save_clustering_metrics(
    #     acc_list=acc_list5,
    #     nmi_list=nmi_list5,
    #     ari_list=ari_list5,
    #     algorithm_name="ap",
    #     dataset_number=file_number
    # )
    # print(f"ap结果已保存到: {file_path}")
    #
    # # 6.用Density-Peak Clustering (DP)聚类标签训练的结果
    # start_time = time.time()
    # pred_custom_dp, acc_list6, nmi_list6, ari_list6 = custom_clustering(X_train, k, X, clustering_method='dp',y=y)
    # time_records['custom_dp'] = time.time() - start_time
    # print("Time of custom_dp:", time_records['custom_dp'])
    # print(f"Using dp accuracy:{cluster_accuracy(y, pred_custom_dp)}")
    # # 调用函数
    # file_path = save_clustering_metrics(
    #     acc_list=acc_list6,
    #     nmi_list=nmi_list6,
    #     ari_list=ari_list6,
    #     algorithm_name="dp",
    #     dataset_number=file_number
    # )
    # print(f"dp结果已保存到: {file_path}")
    #
    # # 7.用Fuzzy C-means聚类标签训练的结果
    # start_time = time.time()
    # pred_custom_fcm, acc_list7, nmi_list7, ari_list7 = custom_clustering(X_train, k, X, clustering_method='fcm',y=y)
    # time_records['custom_fcm'] = time.time() - start_time
    # print("Time of custom_fcm:", time_records['custom_fcm'])
    # print(f"Using fcm accuracy:{cluster_accuracy(y, pred_custom_fcm)}")
    # # 调用函数
    # file_path = save_clustering_metrics(
    #     acc_list=acc_list7,
    #     nmi_list=nmi_list7,
    #     ari_list=ari_list7,
    #     algorithm_name="fcm",
    #     dataset_number=file_number
    # )
    # print(f"fcm结果已保存到: {file_path}")

    # 计算评价指标
    # models = ['Custom_kmeans', 'K-means', 'Custom_hierarchical', 'Agglomerative',
    #           'Custom_spectral', 'Spectral', 'Custom_gmm', 'GMM']
    models = ['Custom_kmeans', 'Custom_hierarchical',
               'Custom_spectral', 'Custom_gmm',
                'Custom_ap', 'Custom_dp',
                'Custom_fcm'
                ]
    accuracies = [
        cluster_accuracy(y, pred_custom),
        # cluster_accuracy(y, pred_custom_hierarchical),
        # cluster_accuracy(y, pred_custom_spectral),
        # cluster_accuracy(y, pred_custom_gmm),
        # cluster_accuracy(y, pred_custom_ap),
        # cluster_accuracy(y, pred_custom_dp),
        # cluster_accuracy(y, pred_custom_fcm),
    ]
    nmi_scores = [
        normalized_mutual_info_score(y, pred_custom),
        # normalized_mutual_info_score(y, pred_custom_hierarchical),
        # normalized_mutual_info_score(y, pred_custom_spectral),
        # normalized_mutual_info_score(y, pred_custom_gmm),
        # normalized_mutual_info_score(y, pred_custom_ap),
        # normalized_mutual_info_score(y, pred_custom_dp),
        # normalized_mutual_info_score(y, pred_custom_fcm),

    ]
    ari_scores = [
        adjusted_rand_score(y, pred_custom),
        # adjusted_rand_score(y, pred_custom_hierarchical),
        # adjusted_rand_score(y, pred_custom_spectral),
        # adjusted_rand_score(y, pred_custom_gmm),
        # adjusted_rand_score(y, pred_custom_ap),
        # adjusted_rand_score(y, pred_custom_dp),
        # adjusted_rand_score(y, pred_custom_fcm),

    ]



    # 打印评价指标对比表格
    df_results = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracies,
        'NMI': nmi_scores,
        'ARI': ari_scores,
    })

    print("\n聚类评价指标对比表格：")
    print(df_results.round(4))

    return df_results


def main():
    all_time_records = []
    #for file_number in range(84, 146):  # 从1到145
    #for file_number in [39, 60, 95, 128, 135, 18, 38, 57, 71, 86, 94, 121, 122, 140, 143, 145]:
    for file_number in [135]:
        # 处理每个数据集
        results = process_data(file_number)

        #all_time_records.append(process_data(file_number))

        if results is None:
            continue  # 如果当前数据集被跳过，继续下一个数据集

        # 处理完results后，如果不再需要，可以删除
        del results
        gc.collect()  # 配合垃圾回收
        print(f"file_number is {file_number}.Experiment completed!")
    print("All files are completed!")

if __name__ == "__main__":
    main()
    #process_data()