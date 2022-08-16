import torch
from fedavg.datasets import get_dataset, VRDataset
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

class Server(object):

    def __init__(self, conf, model, test_df, device):

        self.conf = conf

        self.global_model = model
        self.device = device

        self.test_dataset = get_dataset(conf, test_df)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=conf["batch_size"],shuffle=False)

    def model_aggregate(self, clients_model, weights):

        new_model = {}

        for name, params in self.global_model.state_dict().items():
            new_model[name] = torch.zeros_like(params)

        for key in clients_model.keys():

            for name, param in clients_model[key].items():
                new_model[name]= new_model[name] + clients_model[key][name] * weights[key]

        self.global_model.load_state_dict(new_model)

    @torch.no_grad()
    def model_eval(self):
        self.global_model.eval()

        total_loss = 0.0
        correct = 0
        dataset_size = 0
        predict_prob = []
        labels = []
        
        predict = []

        criterion = torch.nn.CrossEntropyLoss()
        # criterion = torch.nn.functional.cross_entropy()
#         criterion = torch.nn.NLLLoss()
        for batch_id, batch in enumerate(self.test_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda(device=self.device)
                target = target.cuda(device=self.device)

            _, output = self.global_model(data)

            total_loss += criterion(output, target) # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
      

            predict_prob.extend(output.data[:,1].tolist())
#             predict.extend(pred.data.cpu().tolist())
            labels.extend(target.data.cpu().tolist())
          
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

#         多分类任务
        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss.cpu().detach().numpy() / dataset_size

        if self.conf["num_classes"][self.conf["which_dataset"]] == 2:
            roc = roc_auc_score(labels,predict_prob)
        else:
#             roc = roc_auc_score(labels,predict_prob, multi_class='ovr')
            roc = None
            
#         if self.conf["num_classes"][self.conf["which_dataset"]] == 2:
#             f1 = f1_score(labels, predict)
#         else:
#             f1 = f1_score(labels, predict, average='macro')
#         print("roc_auc = {0}, f1_score={1}".format(roc, f1))

        return acc, total_l, roc
