import numpy as np
from scipy.spatial.distance import pdist, squareform


def cluster_dp_k(data, k):
    """
    密度峰值聚类(指定簇数量)

    参数:
    data : ndarray, 形状 (n_samples, n_features)
        输入数据
    k : int
        期望的簇数量

    返回:
    cl : ndarray, 形状 (n_samples,)
        聚类标签
    """
    dist = squareform(pdist(data))
    ND = dist.shape[0]
    dist_line = squareform(dist)
    N = len(dist_line)
    percent = 2.5
    position = round(N * percent / 100)
    sda = np.sort(dist_line)
    dc = sda[position]

    # 计算局部密度(rho)
    rho = np.zeros(ND)
    for i in range(ND - 1):
        for j in range(i + 1, ND):
            if dist[i, j] < dc:
                rho[i] += 1.
                rho[j] += 1.

    maxd = np.max(dist)

    # 计算delta(到更高密度点的最小距离)
    ordrho = np.argsort(-rho)  # 按密度降序排列的索引
    delta = np.zeros(ND)
    nneigh = np.zeros(ND, dtype=int)

    delta[ordrho[0]] = -1.
    nneigh[ordrho[0]] = 0

    for ii in range(1, ND):
        delta[ordrho[ii]] = maxd
        for jj in range(ii):
            if dist[ordrho[ii], ordrho[jj]] < delta[ordrho[ii]]:
                delta[ordrho[ii]] = dist[ordrho[ii], ordrho[jj]]
                nneigh[ordrho[ii]] = ordrho[jj]

    delta[ordrho[0]] = np.max(delta)

    # 计算gamma = rho * delta
    gamma = rho * delta
    orggamma = np.argsort(-gamma)  # 按gamma降序排列的索引

    # 分配到k个簇
    cl = -np.ones(ND, dtype=int)
    Nclu = np.arange(1, k + 1)
    icl = orggamma[:k]
    cl[orggamma[:k]] = Nclu

    # 将剩余点分配到具有更高密度的最近邻的簇
    for i in range(ND):
        if cl[ordrho[i]] == -1:
            cl[ordrho[i]] = cl[nneigh[ordrho[i]]]

    return cl