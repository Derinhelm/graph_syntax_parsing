from copy import deepcopy
from logging import getLogger
import time
import torch
from torch_geometric.utils import scatter
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch

class MLPBlock(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.layers_stack = nn.Sequential(
            nn.Linear(312, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x):
        return self.layers_stack(x)

class MLPNet:
    def __init__(self, options, out_irels_dims, device):
        self.hidden_dims = options["hidden_dims"]
        self.out_irels_dims = out_irels_dims

        self.device = device

        self.unlabeled_res_size = 4 # for a single element in batch
        self.labeled_res_size = 2 * self.out_irels_dims + 2 # for a single element in batch
        self.net = MLPBlock(hidden_channels=self.hidden_dims, \
                            out_channels=self.unlabeled_res_size + \
                                         self.labeled_res_size)
        self.net.to(self.device)

        #self.optimizer = optim.Adam(self.net.parameters(), \
        #                                    lr=options["learning_rate"])
        self.optimizer = optim.SGD(self.net.parameters(),
                                         lr=options["learning_rate"], momentum=0.9)

    def Load(self, epoch):
        mlp_path = 'models/model_mlp' + '_' + str(epoch)

        self.net = MLPBlock(hidden_channels=self.hidden_dims, \
                            out_channels=self.unlabeled_res_size + \
                                         self.labeled_res_size)

        mlp_checkpoint = torch.load(mlp_path)
        self.net.load_state_dict(mlp_checkpoint['model_state_dict'], strict=False)
        self.net.to(self.device)

    def Save(self, epoch):
        info_logger = getLogger('info_logger')
        mlp_path = 'models/model_mlp' + '_' + str(epoch)
        info_logger.info(f'Saving net model to {mlp_path}')
        torch.save({'epoch': epoch, 'model_state_dict': self.net.state_dict()}, \
                   mlp_path)

    def evaluate(self, batch_embeds):
        self.optimizer.zero_grad()
        # batch_embeds - [Tensor]
        batch_embeds = torch.stack(batch_embeds)
        all_scrs_net = self.net(batch_embeds)
        detach_all_scrs = all_scrs_net.clone().detach().cpu()
        return list(all_scrs_net), list(detach_all_scrs)

    def get_scrs_uscrs(self, all_scrs):
        uscrs = all_scrs[:self.unlabeled_res_size]
        scrs = all_scrs[self.unlabeled_res_size:]
        return scrs, uscrs

    def error_processing(self, errs):
        if len(errs) != 0:
            eerrs = torch.sum(torch.stack(errs))
            eerrs.backward()
            self.optimizer.step()
