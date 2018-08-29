import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, num_features, embed_size, hidden=[100], dropout=0.):
        super(GCN, self).__init__()
        self.dropout = dropout

        self.layer_sizes = [num_features] + hidden
        self.layers = []
        # Add all the hidden layers and activation functions
        for i in range(len(self.layer_sizes) - 1):
            self.layers.append(GraphConvolution(self.layer_sizes[i], self.layer_sizes[i + 1]))
            self.layers.append(F.relu(inplace=True))
        # Add the last (embedding) layer with no activation function
        self.layers.append(GraphConvolution(hidden[-1], embed_size))

        # Create the model using nn.Sequential
        self.layers = nn.ModuleList(*layers)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        output = torch.mean(x, dim=1)
        return output


