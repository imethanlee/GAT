import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torchsummary import summary


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, alpha: float, dropout: float):
        super(GraphAttentionLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.w = nn.Parameter(torch.FloatTensor(self.in_channels, self.out_channels))
        self.a = nn.Parameter(torch.FloatTensor(2 * self.out_channels, 1))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
        self.init_params()

    def init_params(self):
        init.xavier_normal_(self.w)
        init.xavier_normal_(self.a)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        wh = torch.matmul(x, self.w)
        num_nodes = wh.shape[0]
        a_input = torch.cat([wh.repeat_interleave(num_nodes, dim=0),
                             wh.repeat(num_nodes, 1)], dim=1)\
            .view(num_nodes, num_nodes, 2 * self.out_channels)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero = -9e15 * torch.ones_like(e)
        attn = torch.where(adj_mat > 0, e, zero)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        h_prime = torch.matmul(attn, wh)
        out = self.elu(h_prime)
        return out


class GAT(nn.Module):
    def __init__(self,
                 n_features: int = 1433,
                 n_hid: int = 8,
                 n_heads: int = 8,
                 n_class: int = 7,
                 dropout: float = 0.6,
                 alpha: float = 0.2):
        super(GAT, self).__init__()
        # 1st-layer multi-head attention
        self.gat1 = nn.ModuleList([GraphAttentionLayer(n_features, n_hid, alpha, dropout) for _ in range(n_heads)])
        self.elu = nn.ELU()
        # 2nd-layer single-head attention
        self.gat2 = GraphAttentionLayer(n_heads * n_hid, n_class, alpha, dropout)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x_do1 = self.dropout(x)
        out1 = torch.cat([attn(x_do1, adj) for attn in self.gat1], dim=1)
        x_do2 = self.dropout(out1)
        out2 = self.gat2(x_do2, adj)
        out = self.log_softmax(out2)
        return out
