import os
import sys
import argparse
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.functional import pairwise_distance
from tensorboardX import SummaryWriter

from .iso_nn_data_util import generate_batch


random_seed = 0
np.random.seed(random_seed)

aggregate_log_dir = "./log"


class SiameseNetwork(nn.Module):
    """
    Standard Siamese network for operation on fixed size graph adjacency matrices.
    """
    def __init__(self, num_nodes):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(num_nodes**2, 1000),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(500, 50),
            # nn.ReLU(inplace=True),
        )

    def forward_single(self, x):
        x = x.view(x.size()[0], -1)
        output = self.fc1(x)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.fc4(output)
        return output

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


def initialise_log_dir(aggregate_log_dir='./log'):
    """Initialises a new directory for the run and returns its path"""
    dirs = os.listdir(aggregate_log_dir)
    previous_runs = list(filter(lambda d: d.startswith('run_'), dirs))
    if len(previous_runs) == 0:
        run_number = 1
    else:
        run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1

    log_dir = f"run_{run_number}"
    return os.path.join(aggregate_log_dir, log_dir)


def accuracy_metrics(distance, label, batch_size):
    num_positive = torch.sum(label)
    num_negative = batch_size - num_positive
    # Average distance between points from the same class
    intraclass_distance = torch.sum(distance * label.squeeze()) / num_positive
    # Average distance between points from different classes
    interclass_distance = torch.sum(distance - distance * label.squeeze()) / num_negative
    return intraclass_distance, interclass_distance


def train(num_nodes, batch_size, num_iter=4000, learning_rate=0.1, directed=False, aggregate_log_dir='./log'):
    # Initiate the TensorBoard logging pipeline
    log_dir = initialise_log_dir(aggregate_log_dir)
    writer = SummaryWriter(log_dir)

    net = SiameseNetwork(num_nodes)
    criterion = ContrastiveLoss()
    optimiser = optim.Adam(net.parameters(), lr=learning_rate)

    counter = []
    loss_history = []
    running_loss = 0.0
    log_every = 100

    for i in range(0, num_iter):
        graph0, graph1, label = generate_batch(batch_size, num_nodes, directed)
        graph0, graph1, label = torch.from_numpy(graph0), torch.from_numpy(graph1), torch.from_numpy(label)
        # Compute the embeddings
        output1, output2 = net(graph0, graph1)
        # Compute the distance between the embeddings
        distance = pairwise_distance(output1, output2)

        optimiser.zero_grad()
        loss = criterion(distance, label)
        loss.backward()
        optimiser.step()

        # Evaluation metrics
        intraclass_distance, interclass_distance = accuracy_metrics(distance, label, batch_size)
        # Log metrics to TensorBoard
        writer.add_scalar('train/loss', loss, i)
        writer.add_scalar('train/intraclass_distance', intraclass_distance, i)
        writer.add_scalar('train/interclass_distance', interclass_distance, i)

        running_loss += loss.item()
        if i % log_every == log_every - 1:
            print(f"Current iteration: {i + 1}\n Loss {running_loss / log_every}\n"
                  f" Batch intraclass distance: {intraclass_distance:6f} |"
                  f" Batch interclass distance: {interclass_distance:6f}")
            # for name, data in net.named_parameters():
            #     print(type(name), type(data))
            #     print(f"iter {i}", param)
            counter.append(i)
            loss_history.append(running_loss / log_every)
            running_loss = 0.0

    writer.close()
    # show_plot(counter, loss_history)

    return counter, loss_history

if __name__ == '__main__':
    # Hyperparameters
    num_nodes = 4
    batch_size = 40  # Must be divisible by 4

    train(num_nodes, batch_size)

