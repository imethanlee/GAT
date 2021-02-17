import numpy as np
import pandas
import scipy.sparse as sp
import torch
from torch.utils.data import TensorDataset


class CoraDataset:
    def __init__(self):
        self.x = pandas.read_pickle("./data/trans.cora.x").toarray()
        self.y = pandas.read_pickle("./data/trans.cora.y")
        self.tx = pandas.read_pickle("./data/trans.cora.tx").toarray()
        self.ty = pandas.read_pickle("./data/trans.cora.ty")
        self.__graph = pandas.read_pickle("./data/trans.cora.graph")
        self.num_nodes = len(self.__graph.keys())
        self.ds_adj_mat = np.zeros((self.num_nodes, self.num_nodes))
        self.preprocessing()

    def preprocessing(self):
        for src in self.__graph.keys():
            items = self.__graph[src]
            for dst in items:
                self.ds_adj_mat[src, dst] = 1

    @property
    def sp_adj_mat(self):
        r, c, v = [], [], []
        for i in range(len(self.ds_adj_mat)):
            for j in range(len(self.ds_adj_mat)):
                if self.ds_adj_mat[i, j] != 0:
                    r.append(i)
                    c.append(j)
                    v.append(float(self.ds_adj_mat[i, j]))
        index = torch.LongTensor([r, c])
        value = torch.FloatTensor(v)
        sp_adj_mat = torch.sparse.FloatTensor(index, value, [self.num_nodes, self.num_nodes])
        return sp_adj_mat

