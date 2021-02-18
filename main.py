from data.cora import *
from model.gat import *


def val():
    pass


def train():
    pass


def test():
    pass


train()
test()

# print(CoraDataset().sp_adj_mat)
#
# tensor = torch.Tensor([[1, 10], [2, 20], [3, 30]])
# print(tensor.repeat(1, 2).view(6, -1))

# a = torch.randn(10,1,2,3)
# b = torch.randn(10,1,2,3)
# print(torch.cat([a, b], dim=1).shape)

adj, features, labels, idx_train, idx_val, idx_test = CoraDataset.load_data()

layer = GraphAttentionLayer(1433, 64).cuda()
layer(features.cuda(), adj.cuda())
