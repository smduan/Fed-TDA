import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, n_feature, n_hidden, n_output, dropout=0.5):
        super(MLP, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_1 = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.bn1 = torch.nn.BatchNorm1d(n_hidden)

        self.hidden_2 = torch.nn.Linear(n_hidden, n_hidden // 2)
        self.bn2 = torch.nn.BatchNorm1d(n_hidden // 2)

        self.hidden_3 = torch.nn.Linear(n_hidden // 2, n_hidden // 4)  # hidden layer
        self.bn3 = torch.nn.BatchNorm1d(n_hidden // 4)

        self.hidden_4 = torch.nn.Linear(n_hidden // 4, n_hidden // 8)  # hidden layer
        self.bn4 = torch.nn.BatchNorm1d(n_hidden // 8)

        self.out = torch.nn.Linear(n_hidden // 8, n_output)  # output layer

    def forward(self, X):
        x = X.view(X.shape[0], -1)

        x = F.relu(self.hidden_1(x))  # hidden layer 1
        x = self.dropout(self.bn1(x))

        x = F.relu(self.hidden_2(x))  # hidden layer 2
        x = self.dropout(self.bn2(x))

        x = F.relu(self.hidden_3(x))  # hidden layer 3
        x = self.dropout(self.bn3(x))

        x = F.relu(self.hidden_4(x))  # hidden layer 4
        feature = self.dropout(self.bn4(x))

        x = self.out(feature)

        return feature, x


class CNN_Model(nn.Module):

    def __init__(self):
        super(CNN_Model, self).__init__()
        self.cnn = nn.Sequential(
            ## Layer 1
            nn.Conv2d(3, 6, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            ## Layer 2
            nn.Conv2d(6, 16, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(

            ##Layer 3
            nn.Linear(400,120),
            nn.ReLU(),

            #Layer 4
            nn.Linear(120,84),
            nn.ReLU(),

            #Layer 5
            nn.Linear(84,84),
            nn.ReLU(),

            #Layer 6
            nn.Linear(84,256)
        )

        #Layer 7  classifier layer
        self.classifier = nn.Linear(256,10)

    def forward(self, input):
        x = self.cnn(input)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, self.classifier(x)

class ReTrainModel(nn.Module):

    def __init__(self):
        super(ReTrainModel, self).__init__()

        #Layer 7  classifier layer
        self.classifier = nn.Linear(256,10)

    def forward(self, input):

        return self.classifier(input)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal(m.weight, 1.0, 0.02)
        torch.nn.init.constant(m.bias, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal(m.weight, 0.0, 0.01)
        torch.nn.init.constant(m.bias, 0.0)




