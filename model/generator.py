import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class fedppd_ConGenerator(nn.Module):

    def __init__(self, noise_dim, feat_dim, out_dim, dropout):
        super(fedppd_ConGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.emb_layer = nn.Embedding(out_dim, out_dim)

        hid_layers = []
        dims = [noise_dim + out_dim, 64, 128, 256]
        for i in range(len(dims) - 1):
            d_in = dims[i]
            d_out = dims[i + 1]
            hid_layers.append(nn.Linear(d_in, d_out))
            hid_layers.append(nn.Tanh())
            hid_layers.append(nn.Dropout(p=dropout, inplace=False))
        self.hid_layers = nn.Sequential(*hid_layers)
        self.nodes_layer = nn.Linear(256, feat_dim)

    def forward(self, z, c):
        z_c = torch.cat((self.emb_layer.forward(c), z), dim=-1) #将噪声拼接到最后一层
        hid = self.hid_layers(z_c)
        node_logits = self.nodes_layer(hid)
        return node_logits