import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class MyTabularDataset(Dataset):

    def __init__(self, dataset, label_col):
        """
        :param dataset: 数据, DataFrame
        :param label_col: 标签列名
        """

        self.label = torch.LongTensor(dataset[label_col].values)

        self.data = dataset.drop(columns=[label_col]).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        label = self.label[index]
        data = self.data[index]

        return torch.tensor(data).float(), label

class MyImageDataset(Dataset):
    def __init__(self, dataset, file_col, label_col):
        """
        :param dataset: 数据， DataFrame
        :param file_col:  文件名称列
        :param label_col:  标签列
        """
        self.file = dataset[file_col].values
        self.label = dataset[label_col].values

        self.normalize = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):

        label = torch.tensor(self.label[index])

        data = Image.open(self.file[index])
        data = self.normalize(data)


        return data, label

class VRDataset(Dataset):
    def __init__(self, data, label):
        """
        :param dataset: 数据， DataFrame
        :param file_col:  文件名称列
        :param label_col:  标签列
        """
        self.data = data

        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        data = torch.tensor(self.data[index])
        label = torch.tensor(self.label[index])

        return data, label

def get_dataset(conf, data):
    """
    :param conf: 配置
    :param data: 数据 (DataFrame)
    :return:
    """
    if conf['data_type'] == 'tabular':
        dataset = MyTabularDataset(data, conf['label_column'])
    elif conf['data_type'] == 'image':
        dataset = MyImageDataset(data, conf['data_column'], conf['label_column'])
    else:
        return None
    return dataset






