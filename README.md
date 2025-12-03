## TabPFN_Clu
````markdown
# TabPFN_Clu: Clustering with Pretrained Tabular Foundation Models

This repository provides the official implementation for the paper:

**Li, Peiwen. “Clustering Method for Tabular Data Based on Foundation Models Pretrained with Synthetic Data.” Computer Research & Development (Data-Centric Computing Special Issue), 2025.**

This project extends **TabPFN v2**, leveraging pretrained tabular foundation models and **nearest-neighbor–aware clustering constraints** to boost structure discovery in tabular datasets.
---

## Environment

```bash
git clone https://github.com/<your_repo>/tabpfn_clu.git
cd tabpfn_clu
pip install -r requirements.txt
````
Python == 3.12 and a CUDA GPU are recommended.

---

## 📁 Project Structure

```
tabpfn/
│
├── data/                     # Demo datasets (.csv/.mat/.data/.txt),given 5 datas(1.data~5.data) for demo testing
├── synthetic_data/           # Synthetic dataset generator
│   ├── synthetic_datasets.py
│   ├── synthetic_datasets1.py
│   └── dataset_plots.py
├── PFN2_2.py                 # used for contrast experiment
├── PFN2_2_finetune.py        # used for paraselect
├── ap_gpt.py                 # implementation of ap clustering
├── dp_implementation.py      # implementation of dp clustering
├── fcm_accuracy.py           # evaluation indicator of fcm clustering
├── gmm_accuracy.py           # evaluation indicator of gmm clustering
├── kmeans_accuracy.py        # evaluation indicator of kmeans clustering
├── spectral_accuracy.py      # evaluation indicator of spectral clustering
├── test_PFN_clustering.py    # demo of contrast experiment
├── test_PFN_paraselect.py    # demo of paraselect
└── requirements.txt          # environment configuration
```

---

## 🚀 Quick Start

### 1. cluster with model

```python
from PFN2_2 import custom_clustering
pred_custom_ap, acc_list5, nmi_list5, ari_list5 = custom_clustering(X_train, k, X, clustering_method='ap',y=y) # based on ap clustering
```
### 2. conduct contrast experiment

```python
run test_PFN_clustering.py
```
### 3. conduct parameter selection

```python
run test_PFN_paraselect.py
```
---

## 🧪 Generate Synthetic Datasets

```python
run synthetic_datasets.py # generate gaussian-like clusters
run synthetic_datasets1.py # generate multiple types of clusters
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
