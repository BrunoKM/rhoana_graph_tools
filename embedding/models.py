import torch
import torch.nn as nn
from embedding_models import GCN


class SiameseNetwork(nn.Module):
    """
    Standard Siamese network for operation on fixed size graph adjacency matrices.
    """
    def __init__(self, num_features, embed_size, sister_net=GCN):
        super(SiameseNetwork, self).__init__()
        self.sister_net = sister_net(num_features, embed_size)

    def forward_single(self, adj):
        x = torch.sum(adj, dim=1, keepdim=True)
        return self.sister_net.forward(x, adj)

    def forward(self, input1, input2):
        output1 = self.forward_single(input1)
        output2 = self.forward_single(input2)
        return output1, output2


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Loss is proportional to square distance when inputs are of the same type, and proportional to
    the square of margin - distance when the classes are different. Margin is a user-specifiable
    hyperparameter.
    """

    def __init__(self, margin=2.0, pos_weight=0.5):
        super(ContrastiveLoss, self).__init__()
        self.pos_weight = pos_weight
        self.margin = margin

    def forward(self, distance, label):
        contrastive_loss = torch.mean((1 - self.pos_weight) * label * torch.pow(distance, 2) +
                                      self.pos_weight * (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return contrastive_loss
