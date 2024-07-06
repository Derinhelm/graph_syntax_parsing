import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, hid2_dim, out_dim, activation):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
                            nn.Linear(in_dim, hid_dim), 
                            activation(),
                            nn.Linear(hid_dim, hid2_dim), 
                            activation(),
                            nn.Linear(hid2_dim, out_dim)
                           )
        self.activation = activation

    def __call__(self,x):
        return self.model(x) # TODO: нужен ли dropout?
