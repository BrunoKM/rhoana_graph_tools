import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn.functional import pairwise_distance
from tensorboardX import SummaryWriter
from .iso_nn_data_util import generate_batch
from .iso_nn_data_util import generate_example
from .models import SiameseNetwork, ContrastiveLoss
from .embedding_models import GCN


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


def train(num_nodes, batch_size=1, embed_size=10, num_iter=4000, learning_rate=0.1, directed=False, aggregate_log_dir='./log'):
    # Initiate the TensorBoard logging pipeline
    log_dir = initialise_log_dir(aggregate_log_dir)
    writer = SummaryWriter(log_dir)

    net = SiameseNetwork(1, embed_size)
    criterion = ContrastiveLoss()
    optimiser = optim.Adam(net.parameters(), lr=learning_rate)

    counter = []
    loss_history = []
    running_loss = 0.0
    log_every = 100

    for i in range(0, num_iter):
        graph0, graph1, label = generate_example(num_nodes, directed)
        graph0, graph1, label = torch.from_numpy(graph0), torch.from_numpy(graph1), torch.from_numpy(label)
        # if batch_size == 1:
        #     graph0 = torch.squeeze(graph0)
        #     graph1 = torch.squeeze(graph1)
        # Compute the embeddings
        output1, output2 = net(graph0, graph1)
        # Compute the distance between the embeddings
        distance = pairwise_distance(torch.unsqueeze(output1, 0), torch.unsqueeze(output2, 0))

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
            counter.append(i)
            loss_history.append(running_loss / log_every)
            running_loss = 0.0

    writer.close()
    # show_plot(counter, loss_history)

    return counter, loss_history

if __name__ == '__main__':
    # Hyperparameters
    num_nodes = 50
    batch_size = 30  # Must be divisible by 4

    train(num_nodes, batch_size)
