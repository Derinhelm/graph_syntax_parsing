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
        self.labeled_res_size = 2 * self.out_irels_dims + 2 # for a single element in batch
        self.gnn = GNNBlock(hidden_channels=self.hidden_dims, \
                            out_channels=self.unlabeled_res_size + \
                                         self.labeled_res_size)
        self.gnn = to_hetero(self.gnn, self.metadata, aggr='sum')
        self.gnn.to(self.device)

        self.optimizer = optim.Adam(self.gnn.parameters(), \
                                            lr=options["learning_rate"])

    def Load(self, epoch):
        gnn_path = 'models/model_gnn' + '_' + str(epoch)

        self.gnn = GNNBlock(hidden_channels=self.hidden_dims, \
                            out_channels=self.unlabeled_res_size + \
                                         self.labeled_res_size)
        
        gnn_checkpoint = torch.load(gnn_path)
        self.gnn.load_state_dict(gnn_checkpoint['model_state_dict'], strict=False)
        self.gnn = to_hetero(self.gnn, self.metadata, aggr='sum')
        self.gnn.to(self.device)

    def Save(self, epoch):
        info_logger = getLogger('info_logger')
        gnn_path = 'models/model_gnn' + '_' + str(epoch)
        info_logger.info(f'Saving gnn model to {gnn_path}')
        torch.save({'epoch': epoch, 'model_state_dict': self.gnn.state_dict()}, \
                   gnn_path)

    def evaluate(self, graph):
        self.optimizer.zero_grad()

        graph_info = graph.x_dict, graph.edge_index_dict
        all_scrs_net = self.gnn(*graph_info)
        all_scrs_clone = all_scrs_net['node'].clone()
        all_scrs_sum = scatter(all_scrs_clone, graph['node'].batch, dim=0, reduce='mean')
        all_scrs = all_scrs_sum.detach().cpu()
        return list(all_scrs)

    def get_scrs_uscrs(self, all_scrs):
        uscrs = all_scrs[:self.unlabeled_res_size]
        scrs = all_scrs[self.unlabeled_res_size:]
        return scrs, uscrs

    def error_processing(self, errs):
        eerrs = torch.sum(torch.tensor(errs, requires_grad=True))
        eerrs.backward()
        self.optimizer.step()
