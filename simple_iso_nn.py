import os
import sys
import argparse
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from iso_nn_data_util import generate_batch


random_seed = 0
np.random.seed(random_seed)

aggregate_log_dir = "./log"


class SiameseNetwork(nn.Module):
    def __init__(self, num_nodes):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(num_nodes**2, 100),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(100, 30),
            # nn.ReLU(inplace=True),
        )

    def forward_single(self, x):
        x = x.view(x.size()[0], -1)
        output = self.fc1(x)
        output = self.fc2(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_single(input1)
        output2 = self.forward_single(input2)
        return output1, output2


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        distance = F.pairwise_distance(output1, output2)
        contrastive_loss = torch.mean(label * torch.pow(distance, 2) +
                                      (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
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


def train(num_nodes, batch_size, num_iter=10000, learning_rate=0.01, directed=False):

    net = SiameseNetwork(num_nodes)
    criterion = ContrastiveLoss()
    optimiser = optim.Adam(net.parameters(), lr=learning_rate)

    counter = []
    loss_history = []
    running_loss = 0.0
    log_every = 1000

    for i in range(0, num_iter):
        graph0, graph1, label = generate_batch(batch_size, num_nodes, directed)
        graph0, graph1, label = torch.from_numpy(graph0), torch.from_numpy(graph1), torch.from_numpy(label)
        output1, output2 = net(graph0, graph1)
        optimiser.zero_grad()
        loss = criterion(output1, output2, label)
        loss.backward()
        optimiser.step()

        # todo: Add evaluation metrics like distance to -ve and distance to -ve

        running_loss += loss.item()
        if i % log_every == log_every - 1:
            print(f"Current iteration: {i + 1}\n Loss {running_loss / log_every}\n")
            counter.append(i)
            loss_history.append(running_loss)
            running_loss = 0.0
            # todo: Incorporate summary saving to a file for tensorboard visualisation
    # show_plot(counter, loss_history)

    # # Add the loss and accuracy values as a scalar to summary.
    # tf.summary.scalar('training_accuracy', accuracy)
    # tf.summary.scalar('loss', loss)
    #
    # # Initialise new logging directory for each run
    # log_dir = initialise_log_dir()

    return counter, loss_history

if __name__ == '__main__':
    # Hyperparameters
    num_nodes = 40
    batch_size = 40  # Must be divisible by 4

    train(num_nodes, batch_size)

