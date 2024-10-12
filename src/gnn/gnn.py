from copy import deepcopy
from logging import getLogger
import time
import torch
from torch_geometric.data import Batch
from torch_geometric.utils import scatter
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
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
        self.hidden_dims = options["hidden_dims"]
        self.out_irels_dims = out_irels_dims

        self.device = device

        self.metadata = (['node'], [('node', 'graph', 'node'), ('node', 'stack', 'node'),\
                                     ('node', 'buffer', 'node')])
        self.unlabeled_res_size = 4 # for a single element in batch
        self.labeled_res_size = 2 * self.out_irels_dims + 2 # for a single element in batch
        self.net = GNNBlock(hidden_channels=self.hidden_dims, \
                            out_channels=self.unlabeled_res_size + \
                                         self.labeled_res_size)
        self.net = to_hetero(self.net, self.metadata, aggr='sum')
        self.net.to(self.device)

        #self.optimizer = optim.Adam(self.net.parameters(), \
        #                                    lr=options["learning_rate"])
        self.optimizer = optim.SGD(self.net.parameters(),
                                         lr=options["learning_rate"], momentum=0.9)

    def Load(self, epoch):
        gnn_path = 'new_models/model_gnn' + '_' + str(epoch)

        self.net = GNNBlock(hidden_channels=self.hidden_dims, \
                            out_channels=self.unlabeled_res_size + \
                                         self.labeled_res_size)

        gnn_checkpoint = torch.load(gnn_path)
        self.net.load_state_dict(gnn_checkpoint['model_state_dict'], strict=False)
        self.net = to_hetero(self.net, self.metadata, aggr='sum')
        self.net.to(self.device)

    def Save(self, epoch):
        info_logger = getLogger('info_logger')
        gnn_path = 'new_models/model_gnn' + '_' + str(epoch)
        info_logger.info(f'Saving gnn model to {gnn_path}')
        torch.save({'epoch': epoch, 'model_state_dict': self.net.state_dict()}, \
                   gnn_path)

    def evaluate(self, graph_list):
        dl = DataLoader(graph_list, batch_size=len(graph_list), shuffle=False)
        batch_list = list(dl)
        if len(dl) > 1:
            print(f"Error batch len in graph DataLoader:{len(dl)}")
            exit(1) # TODO
        graph_batch = batch_list[0]
        print(f"graph:{graph_batch}")
        self.optimizer.zero_grad()

        graph_info = graph_batch.x_dict, graph_batch.edge_index_dict
        all_scrs_net = self.net(*graph_info)
        all_scrs_clone = all_scrs_net['node']
        all_scrs = scatter(all_scrs_clone, graph_batch['node'].batch, dim=0, reduce='mean')
        detach_all_scrs = all_scrs.clone().detach().cpu()
        return list(all_scrs), list(detach_all_scrs)

    def get_scrs_uscrs(self, all_scrs):
        uscrs = all_scrs[:self.unlabeled_res_size]
        scrs = all_scrs[self.unlabeled_res_size:]
        return scrs, uscrs

    def error_processing(self, errs):
        if len(errs) != 0:
            eerrs = torch.sum(torch.stack(errs))
            eerrs.backward()
            self.optimizer.step()
