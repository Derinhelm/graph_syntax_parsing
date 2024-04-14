from copy import deepcopy
from logging import getLogger
import time
import torch
from torch_geometric.utils import scatter
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
    def __init__(self, options, out_irels_dims, device):
        self.elems_in_batch = options["elems_in_batch"]
        self.hidden_dims = options["hidden_dims"]
        self.out_irels_dims = out_irels_dims

        self.device = device

        self.metadata = (['node'], [('node', 'graph', 'node'), ('node', 'stack', 'node'),\
                                     ('node', 'buffer', 'node')])
        self.unlabeled_res_size = 4 # for a single element in batch
        self.unlabeled_GNN = GNNBlock(hidden_channels=self.hidden_dims, out_channels=\
                                                            self.unlabeled_res_size)
        self.unlabeled_GNN = to_hetero(self.unlabeled_GNN, self.metadata, aggr='sum')
        self.unlabeled_GNN.to(self.device)

        self.labeled_res_size = 2 * self.out_irels_dims + 2 # for a single element in batch
        self.labeled_GNN = GNNBlock(hidden_channels=self.hidden_dims, \
                                    out_channels=self.labeled_res_size)
        self.labeled_GNN = to_hetero(self.labeled_GNN, self.metadata, aggr='sum')
        self.labeled_GNN.to(self.device)

        self.unlabeled_optimizer = optim.Adam(self.unlabeled_GNN.parameters(), \
                                              lr=options["learning_rate"])
        self.labeled_optimizer = optim.Adam(self.labeled_GNN.parameters(), \
                                            lr=options["learning_rate"])

    def Load(self, epoch):
        unlab_path = 'models/model_unlab' + '_' + str(epoch)
        lab_path = 'models/model_lab' + '_' + str(epoch)

        self.unlabeled_GNN = GNNBlock(hidden_channels=self.hidden_dims, out_channels=self.unlabeled_res_size)
        self.labeled_GNN = GNNBlock(hidden_channels=self.hidden_dims, \
                                    out_channels=self.labeled_res_size)

        unlab_checkpoint = torch.load(unlab_path)
        self.unlabeled_GNN.load_state_dict(unlab_checkpoint['model_state_dict'], strict=False)
        
        lab_checkpoint = torch.load(lab_path)
        self.labeled_GNN.load_state_dict(lab_checkpoint['model_state_dict'], strict=False)
        
        self.unlabeled_GNN = to_hetero(self.unlabeled_GNN, self.metadata, aggr='sum')
        self.labeled_GNN = to_hetero(self.labeled_GNN, self.metadata, aggr='sum')

        self.unlabeled_GNN.to(self.device)
        self.labeled_GNN.to(self.device)

    def Save(self, epoch):
        info_logger = getLogger('info_logger')
        unlab_path = 'models/model_unlab' + '_' + str(epoch)
        lab_path = 'models/model_lab' + '_' + str(epoch)
        info_logger.info(f'Saving unlabeled model to {unlab_path}')
        torch.save({'epoch': epoch, 'model_state_dict': self.unlabeled_GNN.state_dict()}, \
                   unlab_path)
        info_logger.info(f'Saving labeled model to {lab_path}')
        torch.save({'epoch': epoch, 'model_state_dict': self.labeled_GNN.state_dict()}, lab_path)

    def evaluate(self, graph):
        self.labeled_optimizer.zero_grad()
        self.unlabeled_optimizer.zero_grad()
        graph_info = graph.x_dict, graph.edge_index_dict
        uscrs = self.unlabeled_GNN(*graph_info)
        uscrs_clone = uscrs['node'].clone()
        uscrs_sum = scatter(uscrs_clone, graph['node'].batch, dim=0, reduce='mean')
        uscrs = uscrs_sum.detach().cpu()

        scrs = self.labeled_GNN(*graph_info)
        scrs_clone = scrs['node'].clone()
        scrs_sum = scatter(scrs_clone, graph['node'].batch, dim=0, reduce='mean')
        scrs = scrs_sum.detach().cpu()
        return list(scrs), list(uscrs)

    def error_processing(self, errs):
        eerrs = torch.sum(torch.tensor(errs, requires_grad=True))
        eerrs.backward()
        self.labeled_optimizer.step() # TODO Какой из оптимизаторов ???
        self.unlabeled_optimizer.step()
