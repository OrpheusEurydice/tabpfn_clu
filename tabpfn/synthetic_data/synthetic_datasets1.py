from sklearn.datasets import make_blobs
from pathlib import Path
import pandas as pd
import numpy as np
import random
from tqdm import tqdm  # <--- 加了 tqdm

# ==== 全局参数范围设定 ====
SAMPLE_RANGE = (100, 3000)
FEATURE_RANGE = (2, 200)
CLASS_RANGE = (2, 10)

# ==== 保存路径 ====
output_dir = Path("test")
output_dir.mkdir(parents=True, exist_ok=True)

# ==== 自定义生成函数 ====
def make_grid_clusters(n_samples, n_features, n_classes):
    grid_size = int(np.ceil(np.sqrt(n_classes)))
    centers = [[i, j] for i in range(grid_size) for j in range(grid_size)][:n_classes]
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.1)
    if n_features > 2:
        X = np.hstack([X, np.random.randn(n_samples, n_features - 2)])
    return X, y

def make_multi_gaussian(n_samples, n_features, n_classes):
    samples_per_cluster = n_samples // n_classes
    X, y = [], []
    for i in range(n_classes):
        center = np.random.uniform(-5, 5, 2)
        cov = np.eye(2) * np.random.uniform(0.05, 0.5)
        xi = np.random.multivariate_normal(center, cov, samples_per_cluster)
        if n_features > 2:
            xi = np.hstack([xi, np.random.randn(samples_per_cluster, n_features - 2)])
        X.append(xi)
        y.extend([i] * samples_per_cluster)
    return np.vstack(X), np.array(y)

def make_elliptical_clusters(n_samples, n_features, n_classes):
    samples_per_cluster = n_samples // n_classes
    X, y = [], []
    for i in range(n_classes):
        angle = np.random.uniform(0, np.pi)
        scale = np.array([[np.random.uniform(0.5, 1.5), 0], [0, np.random.uniform(0.1, 1.0)]])
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        cov = rotation @ scale @ rotation.T
        center = np.random.uniform(-5, 5, 2)
        xi = np.random.multivariate_normal(center, cov, samples_per_cluster)
        if n_features > 2:
            xi = np.hstack([xi, np.random.randn(samples_per_cluster, n_features - 2)])
        X.append(xi)
        y.extend([i] * samples_per_cluster)
    return np.vstack(X), np.array(y)

# ==== 数据生成器 ====
generators = {
    "blobs": lambda n, f, c: make_blobs(n_samples=n, centers=c, n_features=f, cluster_std=0.5),
    "grid_clusters": make_grid_clusters,
    "multi_gaussian": make_multi_gaussian,
    "elliptical_clusters": make_elliptical_clusters
}

# ==== 批量生成数据 ====
total_datasets = 50

for i in tqdm(range(total_datasets), desc="Generating datasets"):  # <--- 加了 tqdm
    method = random.choice(list(generators.keys()))
    try:
        n_samples = random.randint(*SAMPLE_RANGE)
        n_features = random.randint(*FEATURE_RANGE)
        n_classes = random.randint(*CLASS_RANGE)
        X, y = generators[method](n_samples, n_features, n_classes)
        df = pd.DataFrame(X, columns=[f"feature_{j}" for j in range(X.shape[1])])
        df["label"] = y
        df.to_csv(output_dir / f"{i:04d}_{method}.csv", index=False)
    except Exception as e:
        print(f"❌ Error generating dataset {i} with method {method}: {e}")

print(f"✅ 完成！共生成 {total_datasets} 个数据集，保存在 {output_dir}")
