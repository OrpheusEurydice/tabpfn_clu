import numpy as np
from sklearn.neighbors import NearestNeighbors

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances


def optimized_bpc_search(X, target_clusters, k_range=(10, 20), alpha_range=(0.5, 0.9),
                         min_cluster_size=10, max_iter=50, verbose=False):
    """
    自动搜索最佳α值使聚类数量最接近目标值

    参数：
        X : 输入数据 (n_samples, n_features)
        target_clusters : 目标簇数量
        k_range : k的搜索范围 (默认尝试10-20)
        alpha_range : α的搜索范围 (默认0.5-0.9)
        min_cluster_size : 最小簇大小
        max_iter : 最大迭代次数
        verbose : 是否打印搜索过程

    返回：
        best_labels : 最佳聚类结果
        best_alpha : 最优α值
        best_k : 最优k值
        history : 搜索过程记录
    """
    best_diff = float('inf')
    best_labels = None
    best_alpha = None
    best_k = None
    history = []
    if target_clusters >= 10:
        target_clusters = 9

    # 参数网格（可根据需要扩展）
    alpha_values = np.linspace(alpha_range[0], alpha_range[1], 10)
    k_values = range(k_range[0], k_range[1] + 1, 2)

    for k in k_values:
        for alpha in alpha_values:
            labels = border_peeling_clustering(
                X, k=k, alpha=alpha,
                min_cluster_size=min_cluster_size,
                max_iter=max_iter
            )

            n_clusters = len(np.unique(labels[labels != -1]))  # 忽略噪声点

            diff = abs(n_clusters - target_clusters)

            history.append({
                'k': k,
                'alpha': alpha,
                'n_clusters': n_clusters,
                'diff': diff
            })

            if diff < best_diff:
                best_diff = diff
                best_labels = labels
                best_alpha = alpha
                best_k = k

            if verbose:
                print(f"k={k:<2} α={alpha:.2f} → {n_clusters} clusters (diff={diff})")

    # if verbose:
    #     valid_labels = best_labels[best_labels != -1]
    #     valid_X = X[best_labels != -1]
    #     print(f"\nBest: k={best_k} α={best_alpha:.3f} → {len(np.unique(valid_labels))} clusters")
    #     print(f"Kept {len(valid_X)}/{len(X)} points after noise removal")
    #
    #     # 返回去除噪声后的数据和标签
    # valid_mask = best_labels != -1
    # return X[valid_mask], best_labels[valid_mask]
    if verbose:
        print(f"\nBest: k={best_k} α={best_alpha:.3f} → {len(np.unique(best_labels[best_labels != -1]))} clusters")

    return best_labels


# 修改后的BPC函数（带安全检测）
def border_peeling_clustering(X, k=10, alpha=0.7, min_cluster_size=10, max_iter=50):
    n_samples = X.shape[0]
    remaining_indices = np.arange(n_samples)
    labels = np.full(n_samples, -1)
    current_cluster = 0

    # 安全检测
    if n_samples <= 1:
        return labels

    # 动态调整初始k
    initial_k = min(k, n_samples - 1)
    nbrs = NearestNeighbors(n_neighbors=initial_k + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    density = 1.0 / np.mean(distances[:, 1:], axis=1)

    while len(remaining_indices) > 0 and max_iter > 0:
        max_iter -= 1
        sub_X = X[remaining_indices]
        sub_density = density[remaining_indices]
        current_n = len(sub_X)

        # 动态调整当前k
        current_k = min(k, current_n - 1)
        if current_k < 1:
            if current_n >= min_cluster_size:
                labels[remaining_indices] = current_cluster
            break

        sub_nbrs = NearestNeighbors(n_neighbors=current_k + 1).fit(sub_X)
        _, sub_indices = sub_nbrs.kneighbors(sub_X)

        local_max_density = np.array([np.max(sub_density[sub_indices[i]])
                                      for i in range(current_n)])
        is_border = sub_density < alpha * local_max_density

        if not np.any(is_border):
            if current_n >= min_cluster_size:
                labels[remaining_indices] = current_cluster
                current_cluster += 1
            break

        core_indices = remaining_indices[~is_border]
        border_indices = remaining_indices[is_border]

        if len(core_indices) >= min_cluster_size:
            labels[core_indices] = current_cluster
            current_cluster += 1

        remaining_indices = border_indices

    return labels
