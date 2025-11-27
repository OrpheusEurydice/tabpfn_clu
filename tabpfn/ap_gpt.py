import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AffinityPropagation


def apcluster_k(data, k, max_iter=2000, convits=200, damping=0.9, prc=10):
    """
    使用调节 preference 参数的方式，使 Affinity Propagation 聚类逼近目标簇数 k。
    如果聚类数 >10，则强制调整参数，使最终结果 ≤10。
    """
    # 相似度矩阵（负欧几里得距离）
    s = -pairwise_distances(data)
    np.fill_diagonal(s, 0)

    # 初始 preference 范围计算
    dpsim1 = np.max(np.sum(s, axis=1))
    pmax = np.max(s)
    dpsim2 = -np.inf

    n = s.shape[0]
    for j1 in range(n - 1):
        for j2 in range(j1 + 1, n):
            temp = np.sum(np.max(s[:, [j1, j2]], axis=1))
            if temp > dpsim2:
                dpsim2 = temp
    pmin = dpsim1 - dpsim2
    low_pref, high_pref = pmin, pmax
    low_k, high_k = 1, n

    def run_ap(pref):
        af = AffinityPropagation(
            affinity='precomputed',
            preference=pref,
            damping=damping,
            max_iter=max_iter,
            convergence_iter=convits,
            random_state=0
        )
        af.fit(s)
        return af

    # 初步搜索：找到聚类数 ≤k 的 preference
    i = -4
    found = False
    while not found:
        temp_pref = high_pref - 10 ** i * (high_pref - low_pref)
        model = run_ap(temp_pref)
        tmp_k = np.unique(model.labels_).size
        if tmp_k <= k:
            found = True
        elif i == -1:
            tmp_k, temp_pref = low_k, low_pref
            found = True
        else:
            i += 1

    # 如果初步结果 >10，强制调整 high_pref 使聚类数 ≤10
    if tmp_k > 10:
        high_pref = temp_pref
        high_k = tmp_k
        # 重新搜索更低的 preference，使聚类数 ≤10
        i = -4
        found = False
        while not found:
            temp_pref = high_pref - 10 ** i * (high_pref - low_pref)
            model = run_ap(temp_pref)
            tmp_k = np.unique(model.labels_).size
            if tmp_k <= 10:
                found = True
            elif i == -1:
                tmp_k, temp_pref = low_k, low_pref
                found = True
            else:
                i += 1

    # 二分法细调 preference，确保聚类数 ≤10
    if abs(tmp_k - k) / k * 100 > prc:
        low_k = tmp_k
        low_pref = temp_pref
        ntries = 0
        while (abs(tmp_k - k) / k * 100 > prc) and (ntries < 20):
            temp_pref = 0.5 * high_pref + 0.5 * low_pref
            model = run_ap(temp_pref)
            tmp_k = np.unique(model.labels_).size
            if tmp_k > 10:  # 如果 >10，强制降低 preference
                high_pref = temp_pref
                high_k = tmp_k
            elif k > tmp_k:
                low_pref = temp_pref
                low_k = tmp_k
            else:
                high_pref = temp_pref
                high_k = tmp_k

            # 提前退出条件
            if np.isclose(high_pref, low_pref):
                break
            ntries += 1

    return model


def apgpt(data, k):
    """
    主入口函数:返回指定簇数 k 的 Affinity Propagation 聚类标签
    """
    model = apcluster_k(np.array(data), k)

    # ==== 新增后处理修正 ====
    raw_labels = model.labels_
    unique, counts = np.unique(raw_labels, return_counts=True)

    # 强制合并多余类别（当实际簇数 >10 时）
    while len(unique) > 10:  # [0][1]
        # 找到两个最小簇进行合并
        min_clusters = unique[np.argsort(counts)[:2]]
        # 将次小簇合并到最小簇
        raw_labels = np.where(raw_labels == min_clusters[1], min_clusters[0], raw_labels)
        # 更新统计
        unique, counts = np.unique(raw_labels, return_counts=True)

    # 重新编号标签（确保从0开始连续）
    _, relabeled = np.unique(raw_labels, return_inverse=True)
    return relabeled