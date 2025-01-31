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
        self.lstm = nn.LSTM(312 * 3, 312 * 3, bidirectional=True, batch_first=True)
        self.layers_stack = nn.Sequential(
            nn.Linear(312 * 3 * 2, hidden_channels), # 2 - because of biderectional LSTM
            nn.ReLU(), # TODO: связываться с calculate_gain
            nn.Linear(hidden_channels, out_channels),
        )
        
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
        relu_gain = torch.nn.init.calculate_gain("relu")
        for child in self.layers_stack.children():
            if isinstance(child, nn.Linear):
                torch.nn.init.xavier_uniform_(child.weight, gain=relu_gain)
                if child.bias is not None:
                    torch.nn.init.zeros_(child.bias)

    def forward(self, x):
        x_lstm, _ = self.lstm(x)
        return self.layers_stack(x_lstm)

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

        self.optimizer = optim.Adam(self.net.parameters(), \
                                            lr=options["learning_rate"])
        #self.optimizer = optim.SGD(self.net.parameters(),
        #                                 lr=options["learning_rate"], momentum=0.9)

    def Load(self, epoch):
        mlp_path = 'new_models/model_mlp' + '_' + str(epoch)

        self.net = MLPBlock(hidden_channels=self.hidden_dims, \
                            out_channels=self.unlabeled_res_size + \
                                         self.labeled_res_size)

        mlp_checkpoint = torch.load(mlp_path)
        self.net.load_state_dict(mlp_checkpoint['model_state_dict'], strict=False)
        self.net.to(self.device)

    def Save(self, epoch):
        info_logger = getLogger('info_logger')
        mlp_path = 'new_models/model_mlp' + '_' + str(epoch)
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
        #print(errs)
        errs.backward()
        self.optimizer.step()
