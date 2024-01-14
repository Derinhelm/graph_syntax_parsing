from logging import getLogger
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, to_hetero
import torch

class GNNBlock(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(312, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class GNNNet:
    def __init__(self, options, out_irels_dims):
        self.hidden_dims = options["hidden_dims"]
        self.out_irels_dims = out_irels_dims

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
        print("device:", self.device)

        self.metadata = (['node'], [('node', 'graph', 'node'), ('node', 'stack', 'node'),\
                                     ('node', 'buffer', 'node')])
        self.unlabeled_GNN = GNNBlock(hidden_channels=self.hidden_dims, out_channels=4)
        self.unlabeled_GNN = to_hetero(self.unlabeled_GNN, self.metadata, aggr='sum')
        self.unlabeled_GNN.to(self.device)

        self.labeled_GNN = GNNBlock(hidden_channels=self.hidden_dims, \
                                    out_channels=2*self.out_irels_dims+2)
        self.labeled_GNN = to_hetero(self.labeled_GNN, self.metadata, aggr='sum')
        self.labeled_GNN.to(self.device)

        self.unlabeled_optimizer = optim.Adam(self.unlabeled_GNN.parameters(), \
                                              lr=options["learning_rate"])
        self.labeled_optimizer = optim.Adam(self.labeled_GNN.parameters(), \
                                            lr=options["learning_rate"])

    def Load(self, epoch):
        unlab_path = 'models/model_unlab' + '_' + str(epoch)
        lab_path = 'models/model_lab' + '_' + str(epoch)

        self.unlabeled_GNN = GNNBlock(hidden_channels=self.hidden_dims, out_channels=4)
        self.labeled_GNN = GNNBlock(hidden_channels=self.hidden_dims, \
                                    out_channels=2*self.out_irels_dims+2)

        unlab_checkpoint = torch.load(unlab_path)
        self.unlabeled_GNN.load_state_dict(unlab_checkpoint['model_state_dict'], strict=False)

        self.unlabeled_GNN = to_hetero(self.unlabeled_GNN, self.metadata, aggr='sum')
        self.labeled_GNN = to_hetero(self.labeled_GNN, self.metadata, aggr='sum')

        self.unlabeled_GNN.to(self.device)
        self.labeled_GNN.to(self.device)

        lab_checkpoint = torch.load(lab_path)
        self.labeled_GNN.load_state_dict(lab_checkpoint['model_state_dict'], strict=False)

    def Save(self, epoch):
        info_logger = getLogger('info_logger')
        unlab_path = 'models/model_unlab' + '_' + str(epoch)
        lab_path = 'models/model_lab' + '_' + str(epoch)
        info_logger.info(f'Saving unlabeled model to {unlab_path}')
        torch.save({'epoch': epoch, 'model_state_dict': self.unlabeled_GNN.state_dict()}, \
                   unlab_path)
        info_logger.info(f'Saving labeled model to {lab_path}')
        torch.save({'epoch': epoch, 'model_state_dict': self.labeled_GNN.state_dict()}, lab_path)

    def evaluate(self, config):
        time_logger = getLogger('time_logger')
        ts = time.time()
        graph = config.config_to_graph()
        graph.to(self.device)
        time_logger.info(f"Time of config_to_graph: {time.time() - ts}")
        ts = time.time()
        uscrs = self.unlabeled_GNN(graph.x_dict, graph.edge_index_dict)
        uscrs = torch.sum(uscrs['node'].clone().detach().cpu(), dim=0)
        scrs = self.labeled_GNN(graph.x_dict, graph.edge_index_dict)
        scrs = torch.sum(scrs['node'].clone().detach().cpu(), dim=0)
        time_logger.info(f"Time of unlabeled_GNN/labeled_GNN: {time.time() - ts}")
        return scrs, uscrs

    def error_processing(self, errs):
        self.labeled_optimizer.zero_grad()
        self.unlabeled_optimizer.zero_grad()
        eerrs = torch.sum(torch.tensor(errs, requires_grad=True))
        eerrs.backward()
        self.labeled_optimizer.step() # TODO Какой из оптимизаторов ???
        self.unlabeled_optimizer.step()
