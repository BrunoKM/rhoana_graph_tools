import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, num_features, embed_size, dropout=0.):
        super(GCN, self).__init__()
        num_hidden = 100
        self.gc1 = GraphConvolution(num_features, num_hidden)
        self.gc2 = GraphConvolution(num_hidden, embed_size)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        output, _ = torch.mean(x, dim=1)
        return output


