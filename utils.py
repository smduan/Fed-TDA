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

    #离散列one-hot
    data= pd.get_dummies(data, columns=discrete_columns["clinical"])
    
    train_data = data[data['tag']=='train'].drop(columns=['tag'])
    test_data = data[data['tag']=='test'].drop(columns=['tag'])

    return train_data, test_data


def label_skew(data,label,K,n_parties,beta,min_require_size = 10):
    """
    :param data: 数据dataframe
    :param label: 标签列名
    :param K: 标签数
    :param n_parties:参与方数
    :param beta: 狄利克雷参数
    :param min_require_size: 点最小数据量，如果低于这个数字会重新划分，保证每个节点数据量不会过少
    :return: 根据狄利克雷分布划分数据到各个参与方
    """
    y_train = data[label]

    min_size = 0
    partition_all = []
    front = np.array([0])
    N = y_train.shape[0]  # N样本总数
    # return train_datasets, test_dataset, n_input, number_samples
    split_data = {}
    
    print("计算各标签对应的数量")
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
            partition = np.diff(partition)#根据切分点求差值来计算各标签划分数据量
            partition_all.append(partition)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    
    # 根据各节点数据index划分数据
    print("开始划分各节点包含的数据")
    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        split_data[j] = data.iloc[idx_batch[j], :]

    return split_data,partition_all


def get_data():

    ###训练数据
    print("开始读取数据集")
    train_data = pd.read_csv(conf["train_dataset"])
    ##测试集,在Server端测试模型效果
    test_data = pd.read_csv(conf["test_dataset"])
    
#     train_data, test_data = min_max_norm(train_data, test_data)
    print("数据集读取完毕，开始对数据按照标签划分多个节点")
    train_data,partition_all = label_skew(train_data,conf["label_column"],conf["num_classes"],conf["num_parties"],conf["beta"])
    
    
    print("各节点数据划分完成")
#     print(partition_all)
    
#     train_datasets = {}
#     val_datasets = {}
    ##各节点数据量
#     number_samples = {}

#     ##读取数据集,训练数据拆分成训练集和测试集
#     for key in train_data.keys():
#         ##打乱顺序
#         train_dataset = shuffle(train_data[key])

#         val_dataset = train_dataset[:int(len(train_dataset) * conf["split_ratio"])]
#         train_dataset = train_dataset[int(len(train_dataset) * conf["split_ratio"]):]
#         train_datasets[key] = train_dataset
#         val_datasets[key] = val_dataset

#         number_samples[key] = len(train_dataset)   

#     print("数据加载完成!")

    return train_data,  test_data


class FedTSNE:
    def __init__(self, X, random_state: int = 1):
        """
        X: ndarray, shape (n_samples, n_features)
        random_state: int, for reproducible results across multiple function calls.
        """
        self.tsne = TSNE(n_components=2, learning_rate='auto', init='random', random_state=random_state)
        self.X_embedded = self.tsne.fit_transform(X)
        self.colors = np.array([[  0,   0,   0],
                                [128,   0,   0],
                                [  0, 128,   0],
                                [128, 128,   0],
                                [  0,   0, 128],
                                [128,   0, 128],
                                [  0, 128, 128],
                                [128, 128, 128],
                                [ 64,   0,   0],
                                [192,   0,   0],
                                [ 64, 128,   0],
                                [192, 128,   0],
                                [ 64,   0, 128],
                                [192,   0, 128],
                                [ 64, 128, 128],
                                [192, 128, 128],
                                [  0,  64,   0],
                                [128,  64,   0],
                                [  0, 192,   0],
                                [128, 192,   0],
                                [  0,  64, 128]]) / 255.

    def visualize(self, y, title=None, save_path='./visualize/tsne.png'):
        assert y.shape[0] == self.X_embedded.shape[0]
        fig, ax = plt.subplots(1, 1)
        ax.scatter(self.X_embedded[:, 0], self.X_embedded[:, 1], c=self.colors[y], s=10)
        ax.set_title(title)
        ax.axis('equal')
        fig.savefig(save_path)
        plt.close(fig)
    
    def visualize_3(self, y_true, y_before, y_after, figsize=None, save_path='./visualize/tsne.png'):
        assert y_true.shape[0] == y_before.shape[0] == y_after.shape[0] == self.X_embedded.shape[0]
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        ax[0].scatter(self.X_embedded[:, 0], self.X_embedded[:, 1], s=2, c=self.colors[y_true])
        ax[1].scatter(self.X_embedded[:, 0], self.X_embedded[:, 1], s=2, c=self.colors[y_before])
        ax[2].scatter(self.X_embedded[:, 0], self.X_embedded[:, 1], s=2, c=self.colors[y_after])
        ax[0].set_title('ground truth')
        ax[1].set_title('before calibration')
        ax[2].set_title('after calibration')
        ax[0].axis('equal')
        ax[1].axis('equal')
        ax[2].axis('equal')
        fig.savefig(save_path)
        plt.close(fig)
