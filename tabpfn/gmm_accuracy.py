import time

from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import contingency_matrix

def cluster_accuracy(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)
    return np.max(cm, axis=0).sum() / np.sum(cm)

def gmm_accuracy(X, y, k, num_runs=10):
    accuracies = []
    NMI = []
    ARI = []
    sum_time = 0

    for run in range(num_runs):
        print(f"Running GMM with seed {run + 1}...")
        start_time = time.time()
        gmm = GaussianMixture(n_components=k, random_state=run)
        pred_gmm = gmm.fit_predict(X)
        one_time = time.time() - start_time
        sum_time += one_time
        print("Time of gmm:", one_time)

        acc = cluster_accuracy(y, pred_gmm)
        nmi = normalized_mutual_info_score(y, pred_gmm)
        ari = adjusted_rand_score(y, pred_gmm)
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
    print(f"Mean time: {mean_time:.4f}")

    final_acc = mean_acc + var_acc
    final_NMI = mean_NMI + var_NMI
    final_ARI = mean_ARI + var_ARI
    return final_acc, final_NMI, final_ARI, mean_time
