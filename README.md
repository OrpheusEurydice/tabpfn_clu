````markdown

## TabPFN_clu

# TabPFN_Clu: Clustering with Pretrained Tabular Foundation Models

This repository provides the official implementation for the paper:

**Li, Peiwen. “Clustering Method for Tabular Data Based on Foundation Models Pretrained with Synthetic Data.” Computer Research & Development (Data-Centric Computing Special Issue), 2025.**

This project extends **TabPFN v2**, leveraging pretrained tabular foundation models and **nearest-neighbor–aware clustering constraints** to boost structure discovery in tabular datasets. It provides a complete pipeline including feature extraction, clustering, baseline evaluation, synthetic data generation, and visualization.

---

## 🔥 Key Features

- **TabPFN v2 embeddings** with GPU support
- **Neighbor-aware clustering constraint**
- **Unified baseline evaluation** (XGBoost, MLP, Logistic Regression, KNN)
- **OpenML dataset loader**
- **Synthetic dataset generation module**
- **Visualization tools** (t-SNE, cluster plots)
- **Full reproducibility aligned with the published paper**

---

## 📦 Installation

```bash
git clone https://github.com/<your_repo>/tabpfn_clu.git
cd tabpfn_clu
pip install -r requirements.txt
````

Python ≥ 3.10 and a CUDA GPU are recommended.

---

## 📁 Project Structure

```
tabpfn/
│
├── data/                     # Demo datasets (.csv/.mat/.data)
├── synthetic_data/           # Synthetic dataset generator
│   ├── synthetic_datasets.py
│   ├── synthetic_datasets1.py
│   └── dataset_plots.py
│
├── test_PFN.py               # TabPFN embedding extraction demo
├── test_PFN_clustering.py    # Clustering with neighbor constraint
├── test_PFN_plot.py          # Embedding visualization
├── test_baseline.py          # Baseline evaluation
├── utils.py                  # KNN, preprocessing, metrics
└── write.py                  # Logging utility
```

---

## 🚀 Quick Start

### 1. Extract TabPFN Embeddings

```python
from tabpfn import extract_pfn_embeddings

X_emb = extract_pfn_embeddings(X_raw)
```

### 2. Cluster with Neighbor Constraints

```python
from tabpfn import train_cluster_model

model = train_cluster_model(X_emb, k_neighbors=20)
labels = model.predict(X_emb)
```

### 3. Evaluate Clustering

```python
from tabpfn import evaluate_clustering

ari, nmi = evaluate_clustering(labels, y_true)
```

---

## 📊 Baseline Evaluation (XGBoost / MLP / Logistic / KNN)

```python
from tabpfn import evaluate_classifier

results = evaluate_classifier(
    embeddings=X_emb,
    labels=y,
    method="mlp",       # options: mlp | xgboost | logistic | knn
    note="TabPFN feature test"
)
print(results)
```

---

## 🧪 Generate Synthetic Datasets

```python
from synthetic_data.synthetic_datasets import generate_synthetic_dataset

X, y = generate_synthetic_dataset(type="moons", noise=0.05)
```

---

## 📈 Visualization (t-SNE)

```python
from tabpfn import plot_embeddings

plot_embeddings(X_emb, labels)
```

---

## 📖 Citation

If you use this repository in your research, please cite:

**Peiwen Li.**
*Clustering Method for Tabular Data Based on Foundation Models Pretrained with Synthetic Data.*
Computer Research & Development, Data-Centric Computing Special Issue, 2025.
DOI: **10.7544/issn1000-1239.202550405**

### BibTeX

```bibtex
@article{Li2025TabPFNCluster,
  title     = {Clustering Method for Tabular Data Based on Foundation Models Pretrained with Synthetic Data},
  author    = {Li, Peiwen},
  journal   = {Computer Research & Development},
  year      = {2025},
  doi       = {10.7544/issn1000-1239.202550405}
}
```
