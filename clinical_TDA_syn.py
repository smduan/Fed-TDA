import pandas as pd
import numpy as np
from collections import Counter
from fedtda.fed_tda import FedTabularDataSyn


def read_train_datasets(path, client_num, label_name):
    """
    读取某个划分好的分散数据集
    path是分散数据集所在的目录
    client_num是划分的数量
    label_name是该数据集的分类列
    分散数据集以"{labelname}_{i}.csv"的形式储存
    """
    train_datasets = []
    for i in range(client_num):
        train_datasets.append(pd.read_csv(path + "{0}_{1}.csv".format(label_name, i)))

    return train_datasets


def get_clients_distribution(datasets, label_name, label_num):
    c = Counter()

    for i in range(len(datasets)):
        c_dataset = Counter(datasets[i][label_name])
        for j in range(label_num):
            c[j] += c_dataset[j]

    return c


def fed_gc_gen_from_path(path, clients_num, label_column, label_num, discrete_columns, privacy_param, out_path, num_row=None):
    # 获取clients数据列表
    train_datasets = read_train_datasets(path, clients_num, label_column)
    distribution = get_clients_distribution(train_datasets, label_column, label_num)

    # 模型拟合
    fed_tda = FedTabularDataSyn()
    fed_tda.fit(train_datasets, discrete_columns=discrete_columns)

    # 数据生成
    sample_class = []
    for i in range(len(distribution)):
        sample_class += [i] * distribution[i]
    # sample_classes = np.array(sample_class)
    # sample_labels = pd.DataFrame({label_column: sample_classes})
    if privacy_param is not None:
        fed_tda.set_privacy(privacy_param[0], privacy_param[1])
    # sample_data = fed_gaussian_copula.sample_remaining_columns(known_columns=sample_labels)

    if num_row is None:
        num_row = 2*sum([len(data) for data in train_datasets])

    sample_data = fed_tda.sample(num_row)
    sample_data.to_csv(out_path, index=False)
    
    
discrete_columns = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking", "label"]

for b in ["b=0.05"]:
    for i in range(5):
        fed_gc_gen_from_path("./data/clinical/{}/".format(b),
                             5,
                             "label",
                             2,
                             discrete_columns,
                             None,
                             "./data/clinical/syn/clinical_syn_{}.csv".format(i),
                             1000)