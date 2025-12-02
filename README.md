TabPFN-Cluster: Clustering with Pretrained Tabular Foundation Models

This repository provides the official implementation for the paper:

Li, Peiwen, Clustering Method for Tabular Data Based on Foundation Models Pretrained with Synthetic Data, Data-Centric Computing, Computer Research & Development, 2025.

This project leverages TabPFN v2 embeddings and introduces nearest-neighbor–aware clustering constraints to enhance structure discovery in tabular data. We provide:

A full pipeline for loading OpenML datasets

TabPFN feature extraction (including GPU acceleration)

Multi-baseline evaluation interfaces (XGBoost, MLP, Logistic Regression, KNN)

Neighbor-constraint learning

Synthetic data generation utilities

Examples and visualization scripts

🔧 Installation
git clone https://github.com/.../tabpfn_clu.git
cd tabpfn_clu
pip install -r requirements.txt


Python ≥ 3.10 and a CUDA-enabled GPU are recommended.

📂 Repository Structure
tabpfn/
│
├── data/                     # Small demo datasets (Flame, UCI data, .mat/.csv)
├── synthetic_data/           # Synthetic dataset generator & visualization
│   ├── synthetic_datasets.py
│   ├── synthetic_datasets1.py
│   └── dataset_plots.py
│
├── test_PFN.py               # Main TabPFN extraction & clustering demo
├── test_PFN_clustering.py    # Neighbor-constraint training example
├── test_PFN_plot.py          # t-SNE / visualization
├── test_baseline.py          # Baseline evaluation (XGBoost / MLP / LR / KNN)
├── utils.py                  # Shared utilities (KNN, preprocessing, metrics)
└── write.py                  # Logging and result writer

▶️ Quick Start
1. Extract TabPFN embeddings
from tabpfn import extract_pfn_embeddings

X_emb = extract_pfn_embeddings(X_raw)

2. Run clustering with neighbor constraints
from tabpfn import train_cluster_model

model = train_cluster_model(X_emb, k_neighbors=20)
labels = model.predict(X_emb)

3. Evaluate with standard metrics
from tabpfn import evaluate_clustering

ari, nmi = evaluate_clustering(labels, y_true)

📊 Baseline Evaluation

You can choose different classifiers for downstream evaluation:

from tabpfn import evaluate_classifier

results = evaluate_classifier(
    embeddings=X_emb,
    labels=y,
    method="mlp",  # mlp | xgboost | logistic | knn
    note="TabPFN feature test"
)

🧪 Synthetic Data Generation
from tabpfn.synthetic_data import generate_synthetic_dataset

X, y = generate_synthetic_dataset(type="moons", noise=0.05)

📖 Cite Our Work

Please cite the following paper if you use this repository:

Peiwen Li.
Clustering Method for Tabular Data Based on Foundation Models Pretrained with Synthetic Data.
Computer Research & Development, Data-Centric Computing Special Issue, 2025.
DOI: 10.7544/issn1000-1239.202550405

BibTeX:

@article{Li2025TabPFNCluster,
  title     = {Clustering Method for Tabular Data Based on Foundation Models Pretrained with Synthetic Data},
  author    = {Li, Peiwen},
  journal   = {Computer Research & Development},
  year      = {2025},
  doi       = {10.7544/issn1000-1239.202550405}
}
