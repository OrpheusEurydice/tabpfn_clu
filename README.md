## TabPFN_Clu
Author: Li Peiwen(202422407020@email.sxu.edu.cn)  
The official implementation for the paper,  
[“Clustering Method for Tabular Data Based on Pretrained Foundation Models with Synthetic Data.”](https://crad.ict.ac.cn/article/doi/10.7544/issn1000-1239.202550405)

## Environment

```bash
git clone https://github.com/<your_repo>/tabpfn_clu.git
cd tabpfn_clu
pip install -r requirements.txt
````
Python == 3.12 and a CUDA GPU are recommended.

---

## 📁 Structure of the repo

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

### 1. cluster with the model

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

Li Peiwen, Li Feijiang, Wang Jieting, Qian Yuhua. Clustering Method for Tabular Data Based on Pretrained Foundation Models with Synthetic Data[J]. Journal of Computer Research and Development, 2025, 62(9): 2139-2151. DOI: 10.7544/issn1000-1239.202550405

### BibTeX

```bibtex
@article{Li2025TabPFNCluster,
  title     = {Clustering Method for Tabular Data Based on Pretrained Foundation Models with Synthetic Data},
  author    = {Li Peiwen, Li Feijiang, Wang Jieting, Qian Yuhua},
  journal   = {Journal of Computer Research and Development},
  volume    = {62},
  year      = {2025},
  doi       = {10.7544/issn1000-1239.202550405}
}
```
