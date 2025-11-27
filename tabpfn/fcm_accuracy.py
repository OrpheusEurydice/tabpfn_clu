import time

import skfuzzy as fuzz
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import contingency_matrix

def cluster_accuracy(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)
    return np.max(cm, axis=0).sum() / np.sum(cm)

def fcm_accuracy(X, y, k, num_runs=10):
    accuracies = []
    NMI = []
    ARI = []
    sum_time = 0

    for run in range(num_runs):
        print(f"Running Fuzzy C-means with seed {run + 1}...")
        # 参数说明：
        # - c: 簇数量
        # - m: 模糊系数(>1)，值越大越模糊
        # - error: 终止误差
        # - maxiter: 最大迭代次数
        start_time = time.time()
        cntr, u, *_ = fuzz.cluster.cmeans(
            X.T, c=k, m=2, error=1e-5, maxiter=1000, seed=run
        )
        pred_fcm = np.argmax(u, axis=0)  # 转换为硬聚类标签
        one_time = time.time() - start_time
        sum_time += one_time
        print("Time of fcm:", one_time)


        acc = cluster_accuracy(y, pred_fcm)
        nmi = normalized_mutual_info_score(y, pred_fcm)
        ari = adjusted_rand_score(y, pred_fcm)
        accuracies.append(acc)
        NMI.append(nmi)
        ARI.append(ari)

    mean_acc = np.mean(accuracies)
    var_acc = np.var(accuracies)
    mean_NMI = np.mean(NMI)
    var_NMI = np.var(NMI)
    mean_ARI = np.mean(ARI)
    var_ARI = np.var(ARI)
    mean_time = sum_time / num_runs

    print(f"Mean accuracy: {mean_acc:.4f}")
    print(f"Accuracy variance: {var_acc:.4f}")

    final_acc = mean_acc + var_acc
    final_NMI = mean_NMI + var_NMI
    final_ARI = mean_ARI + var_ARI
    return final_acc, final_NMI, final_ARI, mean_time
