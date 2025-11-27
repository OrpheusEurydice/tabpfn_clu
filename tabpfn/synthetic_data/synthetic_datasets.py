import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# ----------- 配置区域 ---------------
# 总共生成多少个数据集
NUM_DATASETS = 50

# 样本总数范围（每个数据集）
SAMPLE_RANGE = (100, 600)  # 500到1500个样本

# 特征维度范围
FEATURE_RANGE = (2, 60)  # 2到10维特征

# 类别数量（簇数量）范围
CLASS_RANGE = (2, 8)  # 2到8个簇

# 每个特征的取值范围（均值随机选取）
FEATURE_MEAN_RANGE = (-10, 10)

# 协方差的尺度范围
COV_SCALE_RANGE = (0.2, 2.0)

# 保存目录
SAVE_DIR = "synthetic_datasets"
os.makedirs(SAVE_DIR, exist_ok=True)
# -----------------------------------


def generate_random_gaussian_clusters(n_samples, n_features, n_clusters):
    # 根据簇数量，分配每簇样本数（最后一个簇补足总数）
    samples_per_cluster = np.random.multinomial(n_samples, np.random.dirichlet(np.ones(n_clusters)))

    X = []
    y = []
    for idx in range(n_clusters):
        mean = np.random.uniform(FEATURE_MEAN_RANGE[0], FEATURE_MEAN_RANGE[1], n_features)
        A = np.random.rand(n_features, n_features)
        cov = np.dot(A, A.T)  # 正定矩阵
        scale = np.random.uniform(COV_SCALE_RANGE[0], COV_SCALE_RANGE[1])
        cov *= scale

        cluster_data = np.random.multivariate_normal(mean, cov, samples_per_cluster[idx])
        X.append(cluster_data)
        y.append(np.full(samples_per_cluster[idx], idx))

    X = np.vstack(X)
    y = np.hstack(y)
    return X, y


# 主程序
for i in tqdm(range(NUM_DATASETS), desc="Generating datasets"):
    # 随机选取参数
    n_samples = np.random.randint(SAMPLE_RANGE[0], SAMPLE_RANGE[1] + 1)
    n_features = np.random.randint(FEATURE_RANGE[0], FEATURE_RANGE[1] + 1)
    n_clusters = np.random.randint(CLASS_RANGE[0], CLASS_RANGE[1] + 1)

    # 生成数据
    X, y = generate_random_gaussian_clusters(n_samples, n_features, n_clusters)

    # 保存成CSV
    df = pd.DataFrame(X, columns=[f'feature_{j}' for j in range(n_features)])
    df['label'] = y
    csv_filename = os.path.join(SAVE_DIR, f"dataset_{i:03d}.csv")
    df.to_csv(csv_filename, index=False)

print("✅ 所有数据集生成完毕！保存到 synthetic_datasets/")
