import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torchsummary import summary


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, alpha: float):
        super(GraphAttentionLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.w = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.a = nn.Parameter(torch.FloatTensor(2 * out_channels, 1))
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def init_params(self):
        init.kaiming_uniform_(self.w)
        init.kaiming_uniform_(self.a)

    def forward(self):
        pass


class GAT(nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.gat1 = GraphAttentionLayer()
        self.gat2 = GraphAttentionLayer()
        self.softmax = nn.Softmax()

    def forward(self, x):
        out1 = self.gat1(x)
        out2 = self.gat2(out1)
        out3 = self.softmax(out2)
        return out3