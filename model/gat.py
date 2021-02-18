import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torchsummary import summary
import numpy as np


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, alpha: float = 0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.w = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.a = nn.Parameter(torch.FloatTensor(2 * out_channels, 1))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.softmax = nn.Softmax(dim=1)
        self.elu = nn.ELU()
        self.init_params()

    def init_params(self):
        init.kaiming_uniform_(self.w)
        init.kaiming_uniform_(self.a)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        wh = torch.matmul(x, self.w)
        num_nodes = wh.shape[0]
        a_input = torch.cat([wh.repeat(1, num_nodes).view(num_nodes ** 2, -1),
                             wh.repeat(num_nodes, 1)], dim=1)\
            .view(num_nodes, -1, 2 * self.out_channels)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero = 1e-12 * torch.zeros_like(e)
        attn = torch.where(adj_mat > 0, e, zero)
        attn = self.softmax(attn)
        h_prime = torch.matmul(attn, wh)
        out = self.elu(h_prime)
        return out


class GAT(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int = 7):
        super(GAT, self).__init__()
        self.gat1 = GraphAttentionLayer(in_channels, mid_channels)
        self.gat2 = GraphAttentionLayer(mid_channels, out_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out1 = self.gat1(x)
        out2 = self.gat2(out1)
        out3 = self.softmax(out2)
        return out3