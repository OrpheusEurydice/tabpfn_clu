import os
import re
import shutil
import pandas as pd


def process_round(round_num, template_path):
    # 1. 创建副本文件
    new_excel = f"D:/指定轮数聚类结果(未归一化)/{round_num}轮聚类结果(未归一化).xlsx"
    shutil.copy(template_path, new_excel)

    path = r'D:/所有数据每轮聚类结果(未归一化)'
    # 获取所有条目并筛选出符合格式的文件夹
    folders = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path) and re.match(r'^\d+_results$', entry):
            # 提取开头的数字部分并转换为整数
            num = int(entry.split('_')[0])
            folders.append((num, entry))

    # 按数字从小到大排序
    sorted_folders = sorted(folders, key=lambda x: x[0])

    # 2. 遍历数据集文件夹
    for data_set_index, folder_name in sorted_folders:
        full_dir_path = os.path.join(path, folder_name)
        print(f"指定轮数 {round_num}，处理数据集 {data_set_index} : {folder_name}")

        # 3. 计算Excel行号
        acc_row = 3 + 5 * (data_set_index - 1)
        nmi_row = 4 + 5 * (data_set_index - 1)
        ari_row = 5 + 5 * (data_set_index - 1)

        # 4. 按算法顺序处理文件
        algorithms = ['kmeans', 'hierarchical', 'spectral', 'gmm', 'ap', 'dp', 'fcm']
        for idx, algo in enumerate(algorithms):
            txt_path = os.path.join(full_dir_path, f"ds{data_set_index}_{algo}_metrics.txt")

            # 5. 读取指标数据
            with open(txt_path) as f:
                lines = f.readlines()
                columns = lines[0].split('\t')  # 使用制表符分割
                num_columns = len(columns)  # 当前文件总轮数(列数)

                if num_columns < round_num+1:
                    acc = float(lines[1].split('\t')[num_columns-1])
                    nmi = float(lines[2].split('\t')[num_columns-1])
                    ari = float(lines[3].split('\t')[num_columns-1])
                else:
                    acc = float(lines[1].split('\t')[round_num])
                    nmi = float(lines[2].split('\t')[round_num])
                    ari = float(lines[3].split('\t')[round_num])

            # 6. 确定写入列（3,5,7,9,11,13,15）
            col = 3 + 2 * idx

            # 7. 写入Excel
            df = pd.read_excel(new_excel, header=None)
            df.iloc[acc_row - 1, col - 1] = acc  # Excel从1开始计数
            df.iloc[nmi_row - 1, col - 1] = nmi
            df.iloc[ari_row - 1, col - 1] = ari
            df.to_excel(new_excel, index=False, header=False)


# 遍历所有轮次（示例遍历0-50轮）
for round_num in range(0, 40):
    process_round(round_num, "D:/指定轮数聚类结果(未归一化)/99轮聚类结果(未归一化).xlsx")