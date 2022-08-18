import numpy as np
import pandas as pd
from conf import conf
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def min_max_norm(train_data, test_data):
    train_data['tag'] = "train"
    test_data['tag'] = "test"
    data = pd.concat([train_data, test_data])

    tmp_column = ['label','tag']

    min_max = MinMaxScaler()  
    con = []
    discrete_columns = conf['discrete_columns']
    for c in data.columns:

        if c not in discrete_columns["clinical"] and c not in tmp_column:
            con.append(c)
    print(con)
    data[con] = min_max.fit_transform(data[con])

    data= pd.get_dummies(data, columns=discrete_columns["clinical"])
    
    train_data = data[data['tag']=='train'].drop(columns=['tag'])
    test_data = data[data['tag']=='test'].drop(columns=['tag'])

    return train_data, test_data


def label_skew(data,label,K,n_parties,beta,min_require_size = 10):
    """
    :param data: dataframe
    :param label: 
    :param K: different values in label column
    :param n_parties: num of clients
    :param beta: Dirichlet Param
    :param min_require_size: min num of raw in partition data
    :return: 
    """
    y_train = data[label]

    min_size = 0
    partition_all = []
    front = np.array([0])
    N = y_train.shape[0]
    # return train_datasets, test_dataset, n_input, number_samples
    split_data = {}
    
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            back = np.array([idx_k.shape[0]])
            partition =np.concatenate((front,proportions,back),axis=0)
            partition = np.diff(partition)
            partition_all.append(partition)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    
    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        split_data[j] = data.iloc[idx_batch[j], :]

    return split_data,partition_all


def get_data():

    train_data = pd.read_csv(conf["train_dataset"])
    test_data = pd.read_csv(conf["test_dataset"])
    
#     train_data, test_data = min_max_norm(train_data, test_data)

    train_data,partition_all = label_skew(train_data,conf["label_column"],conf["num_classes"],conf["num_parties"],conf["beta"])
    
#     print(partition_all)
    
#     train_datasets = {}
#     val_datasets = {}
#     number_samples = {}
#     for key in train_data.keys():

#         train_dataset = shuffle(train_data[key])

#         val_dataset = train_dataset[:int(len(train_dataset) * conf["split_ratio"])]
#         train_dataset = train_dataset[int(len(train_dataset) * conf["split_ratio"]):]
#         train_datasets[key] = train_dataset
#         val_datasets[key] = val_dataset

#         number_samples[key] = len(train_dataset)   

#     print("finish to load data.")

    return train_data,  test_data