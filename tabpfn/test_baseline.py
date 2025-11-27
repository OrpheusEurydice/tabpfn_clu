import gc

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tabpfn import TabPFNClassifier

from write import write_test_tree_to_excel


#from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier

# Load data
# X, y = load_breast_cancer(return_X_y=True)
def test_tree(file_number):
    #file_number = 25

    folder = "D:/data_type_data_all"
    file_path = f"{folder}/{file_number}.data"
    print(f"\nProcessing file: {file_path}")

    # 加载数据
    df = pd.read_csv(file_path, delimiter=r'\s+')

    if df.shape[0] > 10000:
        df = df.iloc[:10000]

    y = df.iloc[:, -1].values
    y = np.array(y)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    k = len(np.unique(y))
    # 如果类别数k超过10，则跳过当前数据集
    if k > 10:
        print(f"Skipping file {file_number} as number of clusters (k) is greater than 10.")
        return None

    X = df.iloc[:, :-1].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Initialize a classifier
    clf = TabPFNClassifier()
    clf.fit(X_train, y_train)

    # Predict labels
    predictions = clf.predict(X_test)
    acc_clf = accuracy_score(y_test, predictions)
    print("Accuracy", acc_clf)

    # --- 决策树模型对比 ---
    dt_clf = DecisionTreeClassifier(random_state=42)  # 固定随机种子确保可复现
    dt_clf.fit(X_train, y_train)

    pred_dt = dt_clf.predict(X_test)
    acc_dt = accuracy_score(y_test, pred_dt)
    print("[Decision Tree] Accuracy:", acc_dt)

    write_test_tree_to_excel(acc_clf, acc_dt, data_set_index=file_number)
    return acc_clf

def main():
    for file_number in range(25, 26):  # 从1到145
        # 处理每个数据集
        results = test_tree(file_number)

        if results is None:
            continue  # 如果当前数据集被跳过，继续下一个数据集

        # 处理完results后，如果不再需要，可以删除
        del results
        gc.collect()  # 配合垃圾回收
    print("All files are completed!")

if __name__ == "__main__":
    main()
