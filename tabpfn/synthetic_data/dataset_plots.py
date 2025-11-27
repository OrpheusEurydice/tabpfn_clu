import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm  # 加 tqdm

# 设置路径
dataset_dir = Path("synthetic_datasets")
plot_dir = Path("dataset_plots1")
plot_dir.mkdir(exist_ok=True)

# 遍历所有CSV文件
csv_files = list(dataset_dir.glob("*.csv"))

for csv_file in tqdm(csv_files, desc="Plotting datasets"):  # <<< 加上 tqdm
    try:
        df = pd.read_csv(csv_file)

        if "feature_0" not in df.columns or "feature_1" not in df.columns:
            print(f"跳过 {csv_file.name}：前两列不是 feature_0 和 feature_1")
            continue

        X = df[["feature_0", "feature_1"]].values
        y = df["label"].values if "label" in df.columns else [0] * len(X)

        plt.figure(figsize=(5, 5))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', s=10, alpha=0.8)
        plt.title(csv_file.stem)
        plt.axis('off')

        # 保存图片
        output_path = plot_dir / f"{csv_file.stem}.png"
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    except Exception as e:
        print(f"❌ 绘图失败 {csv_file.name}: {e}")

print(f"✅ 所有图像已保存至 {plot_dir}")
